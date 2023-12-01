//===- FPL22Buffers.h - FPL'22 buffer placement -----------------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPL22BUFFERS_H
#define DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPL22BUFFERS_H

#include "dynamatic/Support/LLVM.h"
#include "dynamatic/Transforms/BufferPlacement/BufferPlacementMILP.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingSupport.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
#include "llvm/ADT/MapVector.h"
#include <set>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

namespace dynamatic {
namespace buffer {
namespace fpl22 {

/// Holds MILP variables associated to every CFDFC unit. Note that a unit may
/// appear in multiple CFDFCs and so may have multiple sets of these variables.
struct UnitVars {
  /// Fluid retiming of tokens at unit's input (real).
  GRBVar retIn;
  /// Fluid retiming of tokens at unit's output. Identical to retiming at unit's
  /// input if the latter is combinational (real).
  GRBVar retOut;
};

/// Holds all MILP variables associated to a channel.
struct ChannelVars {
  llvm::SmallMapVector<SignalType, TimeVars, 4> paths;

  TimeVars elastic;

  GRBVar throughput;

  GRBVar bufPresent;
  GRBVar bufNumSlots;
  llvm::SmallMapVector<SignalType, GRBVar, 4> bufTypePresent;
};

struct BufferPathDelay {
  GRBVar &present;
  double delay;

  BufferPathDelay(GRBVar &present, double delay = 0.0)
      : present(present), delay(delay){};
};

/// Holds all variables associated to a CFDFC union. These are a set of
/// variables for each unit and channel inside the CFDFC union and a CFDFC
/// throughput variable.
// struct CFDFCVars {};

/// Holds all variables that may be used in the MILP. These are a set of
/// variables for each CFDFC and a set of variables for each channel in the
/// function.
struct MILPVars {
  /// Maps each of the CFDFC union's unit to its variables.
  DenseMap<Operation *, UnitVars> units;
  /// Maps each of the CFDFC union's channels to its variables.
  DenseMap<Value, ChannelVars> channels;
  /// CFDFC union's throughputs (real).
  DenseMap<CFDFCUnion *, GRBVar> throughputs;
};

/// Holds the state and logic for FPL22'20 smart buffer placement. To buffer a
/// dataflow circuit, this MILP-based algorithm creates:
/// TODO
class FPL22Buffers : public BufferPlacementMILP {
public:
  /// Target clock period.
  const double targetPeriod;
  /// Maximum clock period.
  const double maxPeriod;

  /// Setups the entire MILP that buffers the input dataflow circuit for the
  /// target clock period, after which (absent errors) it is ready for
  /// optimization. If a channel's buffering properties are provably
  /// unsatisfiable, the MILP status will be set to
  /// `MILPStatus::UNSAT_PROPERTIES` before returning. If something went wrong
  /// during MILP setup, the MILP status will be set to
  /// `MILPStatus::FAILED_TO_SETUP`.
  FPL22Buffers(FuncInfo &funcInfo, const TimingDatabase &timingDB, GRBEnv &env,
               Logger *log = nullptr, double targetPeriod = 4.0,
               double maxPeriod = 8.0);

  /// TODO
  LogicalResult
  getPlacement(DenseMap<Value, PlacementResult> &placement) override;

protected:
  using ChannelFilter = const std::function<bool(Value)> &;

  /// Contains all variables used throughout the MILP.
  MILPVars vars;
  /// All disjoint sets of CFDFC unions, determined from the individual CFDFCs
  /// extracted from the function. Each CFDFC union is made up of all elements
  /// (blocks, units, channels, backedges) that are part of at least one of the
  /// CFDFCs that it was created from. Two CFDFCs end up in the same CFDFC union
  /// if they span over at least one common basic block.
  std::vector<CFDFCUnion> disjointUnions;

  /// Setups the entire MILP, first creating all variables, then all
  /// constraints, and finally setting the system's objective. Called by the
  /// constructor in the absence of prior failures, after which the MILP is
  /// ready to be optimized.
  LogicalResult setup();

  /// Adds all variables used in the MILP to the Gurobi model.
  LogicalResult createVars();

  LogicalResult addCustomChannelConstraints(CFDFCUnion &cfUnion);

  void addChannelPathConstraints(Value channel, SignalType type,
                                 const BufferPathDelay &otherBuffer);

  void addUnitPathConstraints(Operation *unit, SignalType type,
                              ChannelFilter &filter);

  LogicalResult addPathConstraints(CFDFCUnion &cfUnion);

  LogicalResult addElasticityConstraints(CFDFCUnion &cfUnion);

  LogicalResult addThroughputConstraints(CFDFCUnion &cfUnion);

  /// Adds the objective to the Gurobi model.
  LogicalResult addObjective();

  /// Logs placement decisisons and achieved throuhgputs after MILP
  /// optimization. Asserts if the logger is nullptr.
  void logCFDFCUnions();
};

} // namespace fpl22
} // namespace buffer
} // namespace dynamatic
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#endif // DYNAMATIC_TRANSFORMS_BUFFERPLACEMENT_FPL22BUFFERS_H