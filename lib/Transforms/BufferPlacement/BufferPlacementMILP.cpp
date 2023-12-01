//===- BufferPlacementMILP.cpp - MILP-based buffer placement ----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the common MILP-based buffer placement infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;

BufferPlacementMILP::BufferPlacementMILP(FuncInfo &funcInfo,
                                         const TimingDatabase &timingDB,
                                         GRBEnv &env, Logger *logger)
    : MILP<BufferPlacement>(env, logger), timingDB(timingDB),
      funcInfo(funcInfo) {

  // Combines any channel-specific buffering properties coming from IR
  // annotations to internal buffer specifications and stores the combined
  // properties into the channel map. Fails and marks the MILP unsatisfiable if
  // any of those combined buffering properties become unsatisfiable.
  auto deriveBufferingProperties = [&](Channel &channel) -> LogicalResult {
    // Increase the minimum number of slots if internal buffers are present, and
    // check for satisfiability
    if (failed(addInternalBuffers(channel))) {
      unsatisfiable = true;
      std::stringstream ss;
      std::string channelName;
      ss << "Including internal component buffers into buffering "
            "properties of channel '"
         << getUniqueName(*channel.value.getUses().begin())
         << "' made them unsatisfiable. Properties are " << *channel.props;
      if (logger)
        **logger << ss.str();
      return channel.producer->emitError() << ss.str();
    }
    channels[channel.value] = *channel.props;
    return success();
  };

  // Add channels originating from function arguments to the channel map
  for (auto [idx, arg] : llvm::enumerate(funcInfo.funcOp.getArguments())) {
    Channel channel(arg, funcInfo.funcOp, *arg.getUsers().begin());
    if (failed(deriveBufferingProperties(channel)))
      return;
  }

  // Add channels originating from operations' results to the channel map
  for (Operation &op : funcInfo.funcOp.getOps()) {
    for (auto [idx, res] : llvm::enumerate(op.getResults())) {
      Channel channel(res, &op, *res.getUsers().begin());
      if (failed(deriveBufferingProperties(channel)))
        return;
    }
  }

  markReadyToOptimize();
}

LogicalResult
BufferPlacementMILP::addThroughputConstraints(CFDFC &cfdfc,
                                              GRBVar &cfThroughput) {
  // Add a set of constraints for each CFDFC channel
  for (Value channel : cfdfc.channels) {
    // Get the ports the channels connect and their retiming MILP variables
    Operation *srcOp = channel.getDefiningOp();
    Operation *dstOp = *channel.getUsers().begin();
    GRBVar &retSrc = vars.units[srcOp].retOut;
    GRBVar &retDst = vars.units[dstOp].retIn;

    // No throughput constraints on channels going to LSQ stores
    if (isa<handshake::LSQStoreOp>(dstOp))
      continue;

    /// TODO: The legacy implementation does not add any constraints here for
    /// the input channel to select operations that is less frequently
    /// executed. Temporarily, emulate the same behavior obtained from passing
    /// our DOTs to the old buffer pass by assuming the "true" input is always
    /// the least executed one
    if (arith::SelectOp selOp = dyn_cast<arith::SelectOp>(dstOp))
      if (channel == selOp.getTrueValue())
        continue;

    // Retrieve a couple MILP variables associated to the channels
    ChannelVars &chVars = vars.channels[channel];
    GRBVar &bufData = chVars.bufTypePresent[SignalType::DATA];
    GRBVar &bufNumSlots = chVars.bufNumSlots;
    GRBVar &chThroughput = chVars.throughput;
    unsigned backedge = cfdfc.backedges.contains(channel) ? 1 : 0;

    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chThroughput - backedge == retDst - retSrc,
                    "throughput_channelRetiming");
    // If there is an opaque buffer, the CFDFC throughput cannot exceed the
    // channel throughput. If there is not, the CFDFC throughput can exceed
    // the channel thoughput by 1
    model.addConstr(cfThroughput - chThroughput + bufData <= 1,
                    "throughput_cfdfc");
    // If there is an opaque buffer, the summed channel and CFDFC throughputs
    // cannot exceed the number of buffer slots. If there is not, the combined
    // throughput can exceed the number of slots by 1
    model.addConstr(chThroughput + cfThroughput + bufData - bufNumSlots <= 1,
                    "throughput_combined");
    // The channel's throughput cannot exceed the number of buffer slots
    model.addConstr(chThroughput <= bufNumSlots, "throughput_channel");
  }

  // Add a constraint for each pipelined CFDFC union unit
  for (Operation *unit : cfdfc.units) {
    double latency;
    if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)) ||
        latency == 0.0)
      continue;

    // Retrieve the MILP variables corresponding to the unit's fluid retiming
    UnitVars &unitVars = vars.units[unit];
    GRBVar &retIn = unitVars.retIn;
    GRBVar &retOut = unitVars.retOut;

    // The fluid retiming of tokens across the non-combinational unit must
    // be the same as its latency multiplied by the CFDFC union's throughput
    model.addConstr(cfThroughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }
  return success();
}

LogicalResult BufferPlacementMILP::addInternalBuffers(Channel &channel) {
  // Add slots present at the source unit's output ports
  std::string srcName = channel.producer->getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(channel.producer)) {
    channel.props->minTrans += model->outputModel.transparentSlots;
    channel.props->minOpaque += model->outputModel.opaqueSlots;
  }

  // Add slots present at the destination unit's input ports
  std::string dstName = channel.consumer->getName().getStringRef().str();
  if (const TimingModel *model = timingDB.getModel(channel.consumer)) {
    channel.props->minTrans += model->inputModel.transparentSlots;
    channel.props->minOpaque += model->inputModel.opaqueSlots;
  }

  return success(channel.props->isSatisfiable());
}

void BufferPlacementMILP::deductInternalBuffers(Channel &channel,
                                                PlacementResult &result) {
  std::string srcName = channel.producer->getName().getStringRef().str();
  std::string dstName = channel.consumer->getName().getStringRef().str();
  unsigned numTransToDeduct = 0, numOpaqueToDeduct = 0;

  // Remove slots present at the source unit's output ports
  if (const TimingModel *model = timingDB.getModel(channel.producer)) {
    numTransToDeduct += model->outputModel.transparentSlots;
    numOpaqueToDeduct += model->outputModel.opaqueSlots;
  }
  // Remove slots present at the destination unit's input ports
  if (const TimingModel *model = timingDB.getModel(channel.consumer)) {
    numTransToDeduct += model->inputModel.transparentSlots;
    numOpaqueToDeduct += model->inputModel.opaqueSlots;
  }

  assert(result.numTrans >= numTransToDeduct &&
         "not enough transparent slots were placed, the MILP was likely "
         "incorrectly configured");
  assert(result.numOpaque >= numOpaqueToDeduct &&
         "not enough opaque slots were placed, the MILP was likely "
         "incorrectly configured");
  result.numTrans -= numTransToDeduct;
  result.numOpaque -= numOpaqueToDeduct;
}

unsigned BufferPlacementMILP::getChannelNumExecs(Value channel) {
  Operation *srcOp = channel.getDefiningOp();
  if (!srcOp)
    // A channel which originates from a function argument executes only once
    return 1;

  // Iterate over all CFDFCs which contain the channel to determine its total
  // number of executions. Backedges are executed one less time than "forward
  // edges" since they are only taken between executions of the cycle the CFDFC
  // represents
  unsigned numExec = isBackedge(channel) ? 0 : 1;
  for (auto &[cfdfc, _] : funcInfo.cfdfcs)
    if (cfdfc->channels.contains(channel))
      numExec += cfdfc->numExecs;
  return numExec;
}

void BufferPlacementMILP::forEachIOPair(
    Operation *op, const std::function<void(Value, Value)> &callback) {
  for (Value opr : op->getOperands())
    if (!isa<MemRefType>(opr.getType()))
      for (OpResult res : op->getResults())
        if (!isa<MemRefType>(res.getType()))
          callback(opr, res);
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
