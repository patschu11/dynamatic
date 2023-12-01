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
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Support/TimingModels.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "gurobi_c.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetOperations.h"
#include <type_traits>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpl22;

FPL22Buffers::FPL22Buffers(FuncInfo &funcInfo, const TimingDatabase &timingDB,
                           GRBEnv &env, Logger *logger, double targetPeriod,
                           double maxPeriod)
    : BufferPlacementMILP(funcInfo, timingDB, env, logger),
      targetPeriod(targetPeriod), maxPeriod(maxPeriod) {
  if (status == MILPStatus::UNSAT_PROPERTIES)
    return;

  // Create disjoint block unions of all CFDFCs
  SmallVector<CFDFC *, 8> cfdfcs;
  llvm::transform(funcInfo.cfdfcs, std::back_inserter(cfdfcs),
                  [](auto cfAndOpt) { return cfAndOpt.first; });
  getDisjointBlockUnions(cfdfcs, disjointUnions);
  if (logger)
    logCFDFCUnions();

  if (succeeded(setup()))
    status = BufferPlacementMILP::MILPStatus::READY;
}

LogicalResult
FPL22Buffers::getPlacement(DenseMap<Value, PlacementResult> &placement) {
  if (status != MILPStatus::OPTIMIZED) {
    std::stringstream ss;
    ss << status;
    return funcInfo.funcOp->emitError()
           << "Buffer placements cannot be extracted from MILP (reason: "
           << ss.str() << ").";
  }

  return failure();
}

LogicalResult FPL22Buffers::setup() {
  // Create Gurobi variables
  if (failed(createVars()))
    return failure();

  // All constraints are defined per CFDFC union
  for (CFDFCUnion &cfUnion : disjointUnions) {
    if (failed(addCustomChannelConstraints(cfUnion)) ||
        failed(addPathConstraints(cfUnion)) ||
        failed(addElasticityConstraints(cfUnion)) ||
        failed(addThroughputConstraints(cfUnion)))
      return failure();
  }

  return addObjective();
}

LogicalResult FPL22Buffers::createVars() {
  // Create a set of variables for each CFDFC union
  for (auto [uid, cfUnion] : llvm::enumerate(disjointUnions)) {
    std::string prefix = "union" + std::to_string(uid) + "_";

    // Create a Gurobi variable of the given name and type for the CFDFC union
    auto createVar = [&](const std::string &name, char type = GRB_CONTINUOUS) {
      return model.addVar(0, GRB_INFINITY, 0.0, type, name);
    };

    // Create a set of variables for each channel in the CFDFC union
    for (Value cfChannel : cfUnion.channels) {
      // Default-initialize channel variables and retrieve a reference
      ChannelVars &channelVars = vars.channels[cfChannel];
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
      // Variables for throughput constraints
      channelVars.throughput = createChannelVar("throuhgput");
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

    // Create a set of variables for each unit in the CFDFC union
    for (Operation *cfUnit : cfUnion.units) {
      // Default-initialize unit variables and retrieve a reference
      UnitVars &unitVars = vars.units[cfUnit];

      std::string suffix = "_" + getUniqueName(cfUnit);

      // Create a Gurobi variable of the given name and type for the unit
      auto createUnitVar = [&](const std::string &name,
                               char type = GRB_CONTINUOUS) {
        return createVar(name + suffix, type);
      };

      unitVars.retIn = createVar("retIn");

      // If the component is combinational (i.e., 0 latency) its output fluid
      // retiming equals its input fluid retiming, otherwise it is different
      double latency;
      if (failed(timingDB.getLatency(cfUnit, SignalType::DATA, latency)))
        latency = 0.0;
      if (latency == 0.0)
        unitVars.retOut = unitVars.retIn;
      else
        unitVars.retOut = createUnitVar(prefix + "retOut");
    }

    // Create a variable for the CFDFC's throughput
    vars.throughputs[&cfUnion] = createVar(prefix + "throughput");
  }

  // Update the model before returning so that these variables can be referenced
  // safely during the rest of model creation
  model.update();
  return success();
}

LogicalResult FPL22Buffers::addCustomChannelConstraints(CFDFCUnion &cfUnion) {
  for (Value channel : cfUnion.channels) {
    // Get channel-specific buffering properties and channel's variables
    ChannelBufProps &props = channels[channel];
    ChannelVars &chVars = vars.channels[channel];

    // Force buffer presence if at least one slot is requested
    unsigned minSlots = props.minOpaque + props.minTrans;
    if (minSlots > 0)
      model.addConstr(chVars.bufPresent == 1, "custom_forceBuffers");

    // Set constraints based on minimum number of buffer slots
    GRBVar &bufData = chVars.bufTypePresent[SignalType::DATA];
    if (props.minOpaque > 0) {
      // Force the MILP to use opaque slots
      model.addConstr(bufData == 1, "custom_forceData");
      // If the properties ask for both opaque and transparent slots, let
      // opaque slots take over. Transparents slots will be placed "manually"
      // from the total number of slots indicated by the MILP's result.
      model.addConstr(chVars.bufNumSlots >= minSlots, "custom_minData");
    } else if (props.minTrans > 0) {
      // Force the MILP to place a minimum number of transparent slots. If a
      // data buffer is requested, an extra slot must be given for it
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
      }
      // Restrict the maximum number of slots allowed
      model.addConstr(chVars.bufNumSlots <= maxSlots, "custom_maxSlots");
    }

    // Forbid placement of some buffer type based on maximum number of allowed
    // slots on each signal
    if (props.maxOpaque && *props.maxOpaque == 0) {
      // Force the MILP to use transparent slots only
      model.addConstr(bufData == 0, "custom_noData");
    } else if (props.maxTrans && *props.maxTrans == 0) {
      // Force the MILP to use opaque slots only
      GRBVar &bufReady = chVars.bufTypePresent[SignalType::READY];
      model.addConstr(bufReady == 0, "custom_noReady");
    }
  }

  return success();
}

void FPL22Buffers::addChannelPathConstraints(
    Value channel, SignalType type, const BufferPathDelay &otherBuffer) {
  ChannelVars &chVars = vars.channels[channel];
  GRBVar &tIn = chVars.paths[type].tIn;
  GRBVar &tOut = chVars.paths[type].tOut;
  GRBVar &present = chVars.bufTypePresent[type];
  double bigCst = targetPeriod * 10;

  model.addConstr(tIn <= targetPeriod, "path_channelInPeriod");
  model.addConstr(tOut <= targetPeriod, "path_channelOutPeriod");
  model.addConstr(
      tIn - bigCst * present + otherBuffer.delay * otherBuffer.present <= tOut,
      "path_nobuffer");
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

      GRBVar &tInPort = vars.channels[in].paths[type].tOut;
      GRBVar &tOutPort = vars.channels[out].paths[type].tIn;
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

    TimeVars &path = vars.channels[in].paths[type];
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

    TimeVars &path = vars.channels[out].paths[type];
    GRBVar &tOutPort = path.tIn;
    // Arrival time at unit's output port is equal to the output port delay
    model.addConstr(tOutPort == outPortDelay, "path_outDelay");
  }
}

LogicalResult FPL22Buffers::addPathConstraints(CFDFCUnion &cfUnion) {
  // Add path constraints for channels in each timing donain
  for (Value channel : cfUnion.channels) {
    ChannelVars &chVars = vars.channels[channel];
    BufferPathDelay oehb(chVars.bufPresent[SignalType::DATA], 0.1);
    BufferPathDelay tehb(chVars.bufPresent[SignalType::READY], 0.1);
    addChannelPathConstraints(channel, SignalType::DATA, tehb);
    addChannelPathConstraints(channel, SignalType::VALID, tehb);
    addChannelPathConstraints(channel, SignalType::READY, oehb);
  }

  // Add path constraints for units in each timing donain
  for (Operation *unit : cfUnion.units) {
    auto channelFilter = [&](Value channel) -> bool {
      return cfUnion.channels.contains(channel);
    };
    addUnitPathConstraints(unit, SignalType::DATA, channelFilter);
    addUnitPathConstraints(unit, SignalType::VALID, channelFilter);
    addUnitPathConstraints(unit, SignalType::READY, channelFilter);
  }

  return success();
}

LogicalResult FPL22Buffers::addElasticityConstraints(CFDFCUnion &cfUnion) {
  // Upper bound for the longest rigid path
  auto ops = funcInfo.funcOp.getOps();
  unsigned cstCoef = std::distance(ops.begin(), ops.end()) + 2;

  // Add elasticity constraints for channels
  for (Value channel : cfUnion.channels) {
    ChannelVars &chVars = vars.channels[channel];
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
  }

  // Add an elasticity constraint for every input/output port pair in the
  // elastic units
  for (Operation *unit : cfUnion.units) {
    forEachIOPair(unit, [&](Value in, Value out) {
      // The input/output channels must both be inside the CFDFC union
      if (!cfUnion.channels.contains(in) || !cfUnion.channels.contains(out))
        return;

      GRBVar &tInPort = vars.channels[in].elastic.tOut;
      GRBVar &tOutPort = vars.channels[out].elastic.tIn;
      // The elastic arrival time at the output port must be at least one
      // greater than at the input port
      model.addConstr(tOutPort >= 1 + tInPort, "elastic_unitTime");
    });
  }

  return success();
}

LogicalResult FPL22Buffers::addThroughputConstraints(CFDFCUnion &cfUnion) {
  // CFDFC's throughput
  GRBVar &throughput = vars.throughputs[&cfUnion];

  // Add a set of constraints for each CFDFC channel
  for (Value channel : cfUnion.channels) {
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
    unsigned backedge = cfUnion.backedges.contains(channel) ? 1 : 0;

    // If the channel isn't a backedge, its throughput equals the difference
    // between the fluid retiming of tokens at its endpoints. Otherwise, it is
    // one less than this difference
    model.addConstr(chThroughput - backedge == retDst - retSrc,
                    "throughput_channelRetiming");
    // If there is an opaque buffer, the CFDFC throughput cannot exceed the
    // channel throughput. If there is not, the CFDFC throughput can exceed
    // the channel thoughput by 1
    model.addConstr(throughput - chThroughput + bufData <= 1,
                    "throughput_cfdfc");
    // If there is an opaque buffer, the summed channel and CFDFC throughputs
    // cannot exceed the number of buffer slots. If there is not, the combined
    // throughput can exceed the number of slots by 1
    model.addConstr(chThroughput + throughput + bufData - bufNumSlots <= 1,
                    "throughput_combined");
    // The channel's throughput cannot exceed the number of buffer slots
    model.addConstr(chThroughput <= bufNumSlots, "throughput_channel");
  }

  // Add a constraint for each pipelined CFDFC union unit
  for (Operation *unit : cfUnion.units) {
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
    model.addConstr(throughput * latency == retOut - retIn,
                    "through_unitRetiming");
  }
  return success();
}

LogicalResult FPL22Buffers::addObjective() { return success(); }

void FPL22Buffers::logCFDFCUnions() {
  assert(logger && "no logger was provided");
  mlir::raw_indented_ostream &os = **logger;

  // Map each individual CFDFC to its iteration index
  std::map<CFDFC *, size_t> cfIndices;
  for (auto [idx, cfAndOpt] : llvm::enumerate(funcInfo.cfdfcs))
    cfIndices[cfAndOpt.first] = idx;

  os << "# ====================== #\n";
  os << "# Disjoint CFDFCs Unions #\n";
  os << "# ====================== #\n\n";

  // For each CFDFC union, display the blocks it encompasses as well as the
  // individual CFDFCs that fell into it
  for (auto [idx, cfUnion] : llvm::enumerate(disjointUnions)) {

    // Display the blocks making up the union
    auto blockIt = cfUnion.blocks.begin(), blockEnd = cfUnion.blocks.end();
    os << "CFDFC Union #" << idx << ": " << *blockIt;
    while (++blockIt != blockEnd)
      os << ", " << *blockIt;
    os << "\n";

    // Display the block cycle of each CFDFC in the union and some meta
    // information about the union
    os.indent();
    for (CFDFC *cf : cfUnion.cfdfcs) {
      auto cycleIt = cf->cycle.begin(), cycleEnd = cf->cycle.end();
      os << "- CFDFC #" << cfIndices[cf] << ": " << *cycleIt;
      while (++cycleIt != cycleEnd)
        os << " -> " << *cycleIt;
      os << "\n";
    }
    os << "- Number of block: " << cfUnion.blocks.size() << "\n";
    os << "- Number of units: " << cfUnion.units.size() << "\n";
    os << "- Number of channels: " << cfUnion.channels.size() << "\n";
    os << "- Number of backedges: " << cfUnion.backedges.size() << "\n";
    os.unindent();
    os << "\n";
  }
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
