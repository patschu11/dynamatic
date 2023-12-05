//===- FPL22Buffers.cpp - FPL'22 buffer placement ---------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements FPL'22 smart buffer placement.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "gurobi_c.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpl22;

FPL22Buffers::FPL22Buffers(GRBEnv &env, FuncInfo &funcInfo,
                           const TimingDatabase &timingDB, double targetPeriod,
                           CFDFCUnion &cfUnion)
    : BufferPlacementMILP(env, funcInfo, timingDB), targetPeriod(targetPeriod),
      cfUnion(cfUnion) {
  if (succeeded(setup()))
    markReadyToOptimize();
}

FPL22Buffers::FPL22Buffers(GRBEnv &env, FuncInfo &funcInfo,
                           const TimingDatabase &timingDB, double targetPeriod,
                           CFDFCUnion &cfUnion, Logger &logger,
                           StringRef milpName)
    : BufferPlacementMILP(env, funcInfo, timingDB, logger, milpName),
      targetPeriod(targetPeriod), cfUnion(cfUnion) {
  if (succeeded(setup()))
    markReadyToOptimize();
}

void FPL22Buffers::extractResult(BufferPlacement &placement) {
  // Iterate over all channels in the circuit
  for (Value channel : cfUnion.channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    // Extract number and type of slots from the MILP solution, as well as
    // channel-specific buffering properties
    unsigned numSlotsToPlace =
        static_cast<unsigned>(chVars.bufNumSlots.get(GRB_DoubleAttr_X) + 0.5);
    if (numSlotsToPlace == 0)
      continue;

    bool placeOpaque =
        chVars.bufTypePresent[SignalType::DATA].get(GRB_DoubleAttr_X) > 0;
    bool placeTransparent =
        chVars.bufTypePresent[SignalType::READY].get(GRB_DoubleAttr_X) > 0;

    ChannelBufProps &props = channels[channel];
    PlacementResult result;
    if (placeOpaque && placeTransparent) {
      // Place at least one opaque slot and satisfy the opaque slot requirement,
      // all other slots are transparent
      result.numOpaque = std::max(props.minOpaque, 1U);
      result.numTrans = numSlotsToPlace - result.numOpaque;
    } else if (placeOpaque) {
      // Satisfy the transparent slots requirement, all other slots are opaque
      result.numTrans = props.minTrans;
      result.numOpaque = numSlotsToPlace - props.minTrans;
    } else {
      // All slots transparent
      assert(placeTransparent && "slots were placed but of no known type");
      result.numTrans = numSlotsToPlace;
    }

    deductInternalBuffers(channel, result);
    placement[channel] = result;
  }
}

LogicalResult FPL22Buffers::setup() {
  // Create Gurobi variables
  if (failed(createVars()))
    return failure();

  // Add constraints to the MILP
  if (failed(addCustomChannelConstraints()) || failed(addPathConstraints()) ||
      failed(addElasticityConstraints()) || failed(addThroughputConstraints()))
    return failure();

  // Add the MILP objective
  return addObjective();
}

LogicalResult FPL22Buffers::createVars() {
  // Create a Gurobi variable of the given name and type for the CFDFC union
  auto createVar = [&](const std::string &name, char type = GRB_CONTINUOUS) {
    return model.addVar(0, GRB_INFINITY, 0.0, type, name);
  };

  // Create a set of variables for each channel in the CFDFC union
  for (Value cfChannel : cfUnion.channels) {
    // Default-initialize channel variables and retrieve a reference
    ChannelVars &channelVars = vars.channelVars[cfChannel];
    std::string suffix = "_" + getUniqueName(*cfChannel.getUses().begin());

    // Create a Gurobi variable of the given name and type for the channel
    auto createChannelVar = [&](const std::string &name,
                                char type = GRB_CONTINUOUS) {
      return createVar(name + suffix, type);
    };

    // Variables for path constraints
    TimeVars &dataPath = channelVars.paths[SignalType::DATA];
    TimeVars &validPath = channelVars.paths[SignalType::VALID];
    TimeVars &readyPath = channelVars.paths[SignalType::READY];
    dataPath.tIn = createChannelVar("dataPathIn");
    dataPath.tOut = createChannelVar("dataPathOut");
    validPath.tIn = createChannelVar("validPathIn");
    validPath.tOut = createChannelVar("validPathOut");
    readyPath.tIn = createChannelVar("readyPathIn");
    readyPath.tOut = createChannelVar("readyPathOut");
    // Variables for elasticity constraints
    channelVars.elastic.tIn = createChannelVar("elasIn");
    channelVars.elastic.tOut = createChannelVar("elasOut");
    // Variables for placement information
    channelVars.bufPresent = createChannelVar("bufPresent", GRB_BINARY);
    channelVars.bufNumSlots = createChannelVar("bufNumSlots", GRB_INTEGER);
    GRBVar &bufData = channelVars.bufTypePresent[SignalType::DATA];
    GRBVar &bufValid = channelVars.bufTypePresent[SignalType::VALID];
    GRBVar &bufReady = channelVars.bufTypePresent[SignalType::READY];
    bufData = createChannelVar("bufData", GRB_BINARY);
    bufValid = createChannelVar("bufValid", GRB_BINARY);
    bufReady = createChannelVar("bufReady", GRB_BINARY);
  }

  // Create a set of variables for each CFDFC
  for (auto [idx, cf] : llvm::enumerate(cfUnion.cfdfcs)) {
    CFDFCVars &cfVars = vars.cfVars[cf];
    std::string prefix = "cfdfc_" + std::to_string(idx) + "_";

    // Create a variable to represent the throughput of each CFDFC channel
    for (Value channel : cf->channels) {
      cfVars.channelThroughputs[channel] = createVar(
          prefix + "throughput_" + getUniqueName(*channel.getUses().begin()));
    }

    // Create a set of variables for each unit in the CFDFC
    for (Operation *unit : cf->units) {
      std::string suffix = "_" + getUniqueName(unit);

      // Create a Gurobi variable of the given name for the unit
      auto createUnitVar = [&](const std::string &name) {
        return createVar(prefix + name + suffix);
      };

      // Default-initialize unit variables and retrieve a reference
      UnitVars &unitVars = cfVars.unitVars[unit];
      unitVars.retIn = createUnitVar("retIn");

      // If the component is combinational (i.e., 0 latency) its output fluid
      // retiming equals its input fluid retiming, otherwise it is different
      double latency;
      if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)))
        latency = 0.0;
      if (latency == 0.0)
        unitVars.retOut = unitVars.retIn;
      else
        unitVars.retOut = createUnitVar("retOut");
    }

    // Create a variable for the CFDFC's throughput
    cfVars.throughput = createVar(prefix + "throughput");
  }

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
  return success();
}

LogicalResult FPL22Buffers::addCustomChannelConstraints() {
  for (Value channel : cfUnion.channels) {
    // Get channel-specific buffering properties and channel's variables
    ChannelBufProps &props = channels[channel];
    ChannelVars &chVars = vars.channelVars[channel];

    // Force buffer presence if at least one slot is requested
    unsigned minSlots = props.minOpaque + props.minTrans;
    if (minSlots > 0) {
      model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");
      model.addConstr(chVars.bufNumSlots >= minSlots, "custom_minSlots");
    }

    // Set constraints based on minimum number of buffer slots
    GRBVar &bufData = chVars.bufTypePresent[SignalType::DATA];
    GRBVar &bufReady = chVars.bufTypePresent[SignalType::READY];
    if (props.minOpaque > 0) {
      // Force the MILP to place at least one opaque slot
      model.addConstr(bufData == 1, "custom_forceData");
      // If the MILP decides to also place a ready buffer, then we must reserve
      // an extra slot for it
      model.addConstr(chVars.bufNumSlots >= props.minOpaque + bufReady,
                      "custom_minData");
    }
    if (props.minTrans > 0) {
      // Force the MILP to place at least one transparent slot
      model.addConstr(bufReady == 1, "custom_forceReady");
      // If the MILP decides to also place a data buffer, then we must reserve
      // an extra slot for it
      model.addConstr(chVars.bufNumSlots >= props.minTrans + bufData,
                      "custom_minReady");
    }

    // Set constraints based on maximum number of buffer slots
    if (props.maxOpaque && props.maxTrans) {
      unsigned maxSlots = *props.maxTrans + *props.maxOpaque;
      if (maxSlots == 0) {
        // Forbid buffer placement on the channel entirely
        model.addConstr(chVars.bufPresent == 0, "custom_noBuffer");
        model.addConstr(chVars.bufNumSlots == 0, "custom_noSlot");
      } else {
        // Restrict the maximum number of slots allowed
        model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
      }
    }

    // Forbid placement of some buffer type based on maximum number of allowed
    // slots on each signal
    if (props.maxOpaque && *props.maxOpaque == 0) {
      // Force the MILP to use transparent slots only
      model.addConstr(bufData == 0, "custom_noData");
    } else if (props.maxTrans && *props.maxTrans == 0) {
      // Force the MILP to use opaque slots only
      model.addConstr(bufReady == 0, "custom_noReady");
    }
  }

  return success();
}

void FPL22Buffers::addChannelPathConstraints(
    Value channel, SignalType type, const BufferPathDelay &otherBuffer) {
  ChannelVars &chVars = vars.channelVars[channel];
  GRBVar &tIn = chVars.paths[type].tIn;
  GRBVar &tOut = chVars.paths[type].tOut;
  GRBVar &present = chVars.bufTypePresent[type];
  double bigCst = targetPeriod * 10;

  model.addConstr(tIn <= targetPeriod, "path_channelInPeriod");
  model.addConstr(tOut <= targetPeriod, "path_channelOutPeriod");
  model.addConstr(
      tIn - bigCst * present + otherBuffer.delay * otherBuffer.present <= tOut,
      "path_noBuffer");
  model.addConstr(otherBuffer.delay * otherBuffer.present <= tOut,
                  "path_buffer");
}

void FPL22Buffers::addUnitPathConstraints(Operation *unit, SignalType type,
                                          ChannelFilter &filter) {
  // Add path constraints for units
  double latency;
  if (failed(timingDB.getLatency(unit, type, latency)))
    latency = 0.0;

  if (latency == 0.0) {
    double delay;
    if (failed(timingDB.getTotalDelay(unit, type, delay)))
      delay = 0.0;

    // The unit is not pipelined, add a path constraint for each input/output
    // port pair in the unit
    forEachIOPair(unit, [&](Value in, Value out) {
      // The input/output channels must both be inside the CFDFC union
      if (!filter(in) || !filter(out))
        return;

      // Flip channels on ready path which goes upstream
      if (type == SignalType::READY)
        std::swap(in, out);

      GRBVar &tInPort = vars.channelVars[in].paths[type].tOut;
      GRBVar &tOutPort = vars.channelVars[out].paths[type].tIn;
      // Arrival time at unit's output port must be greater than arrival
      // time at unit's input port + the unit's combinational data delay
      model.addConstr(tOutPort >= tInPort + delay, "path_combDelay");
    });

    return;
  }

  // The unit is pipelined, add a constraint for every of the unit's inputs
  // and every of the unit's output ports

  // Input port constraints
  for (Value in : unit->getOperands()) {
    if (!filter(in))
      continue;

    double inPortDelay;
    if (failed(timingDB.getPortDelay(unit, type, PortType::IN, inPortDelay)))
      inPortDelay = 0.0;

    TimeVars &path = vars.channelVars[in].paths[type];
    GRBVar &tInPort = path.tOut;
    // Arrival time at unit's input port + input port delay must be less
    // than the target clock period
    model.addConstr(tInPort + inPortDelay <= targetPeriod, "path_inDelay");
  }

  // Output port constraints
  for (OpResult out : unit->getResults()) {
    if (!filter(out))
      continue;

    double outPortDelay;
    if (failed(timingDB.getPortDelay(unit, type, PortType::OUT, outPortDelay)))
      outPortDelay = 0.0;

    TimeVars &path = vars.channelVars[out].paths[type];
    GRBVar &tOutPort = path.tIn;
    // Arrival time at unit's output port is equal to the output port delay
    model.addConstr(tOutPort == outPortDelay, "path_outDelay");
  }
}

namespace {
struct Pin {
  Value channel;
  SignalType type;

  Pin(Value channel, SignalType type) : channel(channel), type(type){};
};

struct MixedDomainConstraint {
  Pin input;
  Pin output;
  double internalDelay;

  MixedDomainConstraint(Pin input, Pin output, double internalDelay)
      : input(input), output(output), internalDelay(internalDelay){};
};

} // namespace

void FPL22Buffers::addUnitMixedPathConstraints(Operation *unit,
                                               ChannelFilter &filter) {
  std::vector<MixedDomainConstraint> constraints;
  const TimingModel *unitModel = timingDB.getModel(unit);

  // Adds constraints between the input ports' valid and ready pins of a unit
  // with two operands.
  auto addJoinedOprdConstraints = [&]() -> void {
    double vr = unitModel->validToReady;
    Value oprd0 = unit->getOperand(0), oprd1 = unit->getOperand(1);
    constraints.emplace_back(Pin(oprd0, SignalType::VALID),
                             Pin(oprd1, SignalType::READY), vr);
    constraints.emplace_back(Pin(oprd1, SignalType::VALID),
                             Pin(oprd0, SignalType::READY), vr);
  };

  // Adds constraints between the data pin of the provided input channel and all
  // valid/ready output pins.
  auto addDataToAllValidReadyConstraints = [&](Value inputChannel) -> void {
    Pin input(inputChannel, SignalType::DATA);
    double cv = unitModel->condToValid;
    for (OpResult res : unit->getResults())
      constraints.emplace_back(input, Pin(res, SignalType::VALID), cv);
    double cr = unitModel->condToReady;
    for (Value oprd : unit->getOperands())
      constraints.emplace_back(input, Pin(oprd, SignalType::READY), cr);
  };

  llvm::TypeSwitch<Operation *, void>(unit)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condBrOp) {
            // There is a path between the data pin of the condition operand and
            // every valid/ready output pin
            addDataToAllValidReadyConstraints(condBrOp.getConditionOperand());

            // The two branch inputs are joined therefore there are cross
            // connections between the valid and ready pins
            addJoinedOprdConstraints();
          })
      .Case<handshake::ControlMergeOp>([&](handshake::ControlMergeOp cmergeOp) {
        // There is a path between the valid pin of the first operand and the
        // data pin of the index result
        Pin input(cmergeOp.getOperand(0), SignalType::VALID);
        Pin output(cmergeOp.getIndex(), SignalType::DATA);
        constraints.emplace_back(input, output, unitModel->validToCond);
      })
      .Case<handshake::MergeOp>([&](handshake::MergeOp mergeOp) {
        // There is a path between every valid input pin and the data output
        // pin
        double vd = unitModel->validToData;
        Pin output(mergeOp.getResult(), SignalType::DATA);
        for (Value oprd : mergeOp->getOperands())
          constraints.emplace_back(Pin(oprd, SignalType::VALID), output, vd);
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // There is a path between the data pin of the select operand and every
        // valid/ready output pin
        addDataToAllValidReadyConstraints(muxOp.getSelectOperand());

        // There is a path between every valid input pin and every data/ready
        // output pin
        double vd = unitModel->validToData;
        double vr = unitModel->validToReady;
        for (Value oprd : muxOp->getOperands()) {
          for (OpResult res : muxOp->getResults()) {
            constraints.emplace_back(Pin(oprd, SignalType::VALID),
                                     Pin(res, SignalType::DATA), vd);
          }
          for (Value readyOprd : muxOp->getOperands()) {
            constraints.emplace_back(Pin(oprd, SignalType::VALID),
                                     Pin(readyOprd, SignalType::READY), vr);
          }
        }
      })
      .Case<handshake::MCLoadOp, handshake::LSQLoadOp, handshake::MCStoreOp,
            handshake::LSQStoreOp, arith::AddIOp, arith::AddFOp, arith::SubIOp,
            arith::SubFOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
            arith::MulIOp, arith::MulFOp, arith::DivUIOp, arith::DivSIOp,
            arith::DivFOp, arith::SIToFPOp, arith::RemSIOp, arith::ShRSIOp,
            arith::ShLIOp, arith::CmpIOp, arith::CmpFOp>(
          [&](auto) { addJoinedOprdConstraints(); });

  std::string unitName = getUniqueName(unit);
  unsigned idx = 0;
  for (MixedDomainConstraint &cons : constraints) {
    // The input/output channels must both be inside the CFDFC union
    if (!filter(cons.input.channel) || !filter(cons.output.channel))
      return;

    // Find variables for arrival time at input/output pin
    GRBVar &tPinIn =
        vars.channelVars[cons.input.channel].paths[cons.input.type].tOut;
    GRBVar &tPinOut =
        vars.channelVars[cons.output.channel].paths[cons.input.type].tIn;

    // Arrival time at unit's output pin must be greater than arrival time at
    // unit's input pin plus the unit's internal delay on the path
    std::string consName =
        "path_mixed_" + unitName + "_" + std::to_string(idx++);
    model.addConstr(tPinIn + cons.internalDelay <= tPinOut, consName);
  }
}

LogicalResult FPL22Buffers::addPathConstraints() {
  // Add path constraints for channels in each timing donain
  for (Value channel : cfUnion.channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    BufferPathDelay oehb(chVars.bufTypePresent[SignalType::DATA], 0.1);
    BufferPathDelay tehb(chVars.bufTypePresent[SignalType::READY], 0.1);
    addChannelPathConstraints(channel, SignalType::DATA, tehb);
    addChannelPathConstraints(channel, SignalType::VALID, tehb);
    addChannelPathConstraints(channel, SignalType::READY, oehb);
  }

  auto channelFilter = [&](Value channel) -> bool {
    return cfUnion.channels.contains(channel);
  };

  // Add path constraints for units in each timing donain
  for (Operation *unit : cfUnion.units) {
    addUnitPathConstraints(unit, SignalType::DATA, channelFilter);
    addUnitPathConstraints(unit, SignalType::VALID, channelFilter);
    addUnitPathConstraints(unit, SignalType::READY, channelFilter);
  }

  // Add path constraints for units in the mixed domain
  for (Operation *unit : cfUnion.units)
    addUnitMixedPathConstraints(unit, channelFilter);
  return success();
}

LogicalResult FPL22Buffers::addElasticityConstraints() {
  // Upper bound for the longest rigid path
  auto ops = funcInfo.funcOp.getOps();
  unsigned cstCoef = std::distance(ops.begin(), ops.end()) + 2;

  // Add elasticity constraints for channels
  for (Value channel : cfUnion.channels) {
    ChannelVars &chVars = vars.channelVars[channel];
    GRBVar &tIn = chVars.elastic.tIn;
    GRBVar &tOut = chVars.elastic.tOut;
    GRBVar &bufPresent = chVars.bufPresent;
    GRBVar &bufData = chVars.bufTypePresent[SignalType::DATA];
    GRBVar &bufValid = chVars.bufTypePresent[SignalType::VALID];
    GRBVar &bufReady = chVars.bufTypePresent[SignalType::READY];
    GRBVar &bufNumSlots = chVars.bufNumSlots;

    // If there is a data buffer on the channel, the channel elastic
    // arrival time at the ouput must be greater than at the input (breaks
    // cycles!)
    model.addConstr(tOut >= tIn - cstCoef * bufData, "elastic_cycle");
    // There must be enough slots for the data and ready paths
    model.addConstr(bufNumSlots >= bufData + bufReady, "elastic_slots");
    // The number of data and valid slots is the same
    model.addConstr(bufData == bufValid, "elastic_dataValid");
    // If there is at least one slot, there must be a buffer
    model.addConstr(bufPresent >= 0.01 * bufNumSlots, "elastic_present");
    // If there is a buffer present, it must be either on data or ready
    model.addConstr(bufData + bufReady >= bufPresent, "elastic_dataReady");
  }

  // Add an elasticity constraint for every input/output port pair in the
  // elastic units
  for (Operation *unit : cfUnion.units) {
    forEachIOPair(unit, [&](Value in, Value out) {
      // The input/output channels must both be inside the CFDFC union
      if (!cfUnion.channels.contains(in) || !cfUnion.channels.contains(out))
        return;

      GRBVar &tInPort = vars.channelVars[in].elastic.tOut;
      GRBVar &tOutPort = vars.channelVars[out].elastic.tIn;
      // The elastic arrival time at the output port must be at least one
      // greater than at the input port
      model.addConstr(tOutPort >= 1 + tInPort, "elastic_unitTime");
    });
  }

  return success();
}

LogicalResult FPL22Buffers::addThroughputConstraints() {
  // Add a set of constraints for each CFDFC in the union
  for (CFDFC *cf : cfUnion.cfdfcs) {
    CFDFCVars &cfVars = vars.cfVars[cf];

    // Add a set of constraints for each channel in the CFDFC
    for (Value channel : cf->channels) {
      // Get the ports the channels connect and their retiming MILP variables
      Operation *srcOp = channel.getDefiningOp();
      Operation *dstOp = *channel.getUsers().begin();
      GRBVar &retSrc = cfVars.unitVars[srcOp].retOut;
      GRBVar &retDst = cfVars.unitVars[dstOp].retIn;

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
      ChannelVars &chVars = vars.channelVars[channel];
      GRBVar &bufData = chVars.bufTypePresent[SignalType::DATA];
      GRBVar &bufNumSlots = chVars.bufNumSlots;
      GRBVar &chThroughput = cfVars.channelThroughputs[channel];
      unsigned backedge = cf->backedges.contains(channel) ? 1 : 0;

      // If the channel isn't a backedge, its throughput equals the difference
      // between the fluid retiming of tokens at its endpoints. Otherwise, it is
      // one less than this difference
      model.addConstr(chThroughput - backedge == retDst - retSrc,
                      "throughput_channelRetiming");
      // If there is an opaque buffer, the CFDFC throughput cannot exceed the
      // channel throughput. If there is not, the CFDFC throughput can exceed
      // the channel thoughput by 1
      model.addConstr(cfVars.throughput - chThroughput + bufData <= 1,
                      "throughput_cfdfc");
      // If there is an opaque buffer, the summed channel and CFDFC throughputs
      // cannot exceed the number of buffer slots. If there is not, the combined
      // throughput can exceed the number of slots by 1
      model.addConstr(
          chThroughput + cfVars.throughput + bufData - bufNumSlots <= 1,
          "throughput_combined");
      // The channel's throughput cannot exceed the number of buffer slots
      model.addConstr(chThroughput <= bufNumSlots, "throughput_channel");
    }

    // Add a set of constraints for each pipelined unit in the CFDFC
    for (Operation *unit : cf->units) {
      double latency;
      if (failed(timingDB.getLatency(unit, SignalType::DATA, latency)) ||
          latency == 0.0)
        continue;

      // Retrieve the MILP variables corresponding to the unit's fluid retiming
      UnitVars &unitVars = cfVars.unitVars[unit];
      GRBVar &retIn = unitVars.retIn;
      GRBVar &retOut = unitVars.retOut;

      // The fluid retiming of tokens across the non-combinational unit must
      // be the same as its latency multiplied by the CFDFC's throughput
      model.addConstr(cfVars.throughput * latency == retOut - retIn,
                      "through_unitRetiming");
    }
  }
  return success();
}

LogicalResult FPL22Buffers::addObjective() {
  // Compute the total number of executions over all channels
  unsigned totalExecs = 0;
  for (Value channel : cfUnion.channels)
    totalExecs += getChannelNumExecs(channel);

  // Create the expression for the MILP objective
  GRBLinExpr objective;

  // For each CFDFC, add a throughput contribution to the objective, weighted
  // by the "importance" of the CFDFC
  double maxCoefCFDFC = 0.0;
  if (totalExecs != 0) {
    for (CFDFC *cf : cfUnion.cfdfcs) {
      if (!funcInfo.cfdfcs[cf])
        continue;
      double coef =
          cf->channels.size() * cf->numExecs / static_cast<double>(totalExecs);
      objective += coef * vars.cfVars[cf].throughput;
      maxCoefCFDFC = std::max(coef, maxCoefCFDFC);
    }
  }

  // In case we ran the MILP without providing any CFDFC, set the maximum CFDFC
  // coefficient to any positive value
  if (maxCoefCFDFC == 0.0)
    maxCoefCFDFC = 1.0;

  // For each channel, add a "penalty" in case a buffer is added to the channel,
  // and another penalty that depends on the number of slots
  double bufPenaltyMul = 1e-4;
  double slotPenaltyMul = 1e-5;
  for (auto &[channel, chVar] : vars.channelVars) {
    objective -= maxCoefCFDFC * bufPenaltyMul * chVar.bufPresent;
    objective -= maxCoefCFDFC * slotPenaltyMul * chVar.bufNumSlots;
  }

  // Finally, set the MILP objective
  model.setObjective(objective, GRB_MAXIMIZE);
  return success();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
