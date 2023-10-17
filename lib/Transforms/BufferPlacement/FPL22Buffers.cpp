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

#include "llvm/ADT/SetOperations.h"
#include <unordered_set>
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/FPL22Buffers.h"
#include "dynamatic/Transforms/PassDetails.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"

using namespace llvm::sys;
using namespace circt;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::buffer::fpl22;

using BlockSet = std::unordered_set<unsigned>;

FPL22Buffers::FPL22Buffers(FuncInfo &funcInfo, const TimingDatabase &timingDB,
                           GRBEnv &env, Logger *logger, double targetPeriod,
                           double maxPeriod)
    : BufferPlacementMILP(funcInfo, timingDB, env, logger),
      targetPeriod(targetPeriod), maxPeriod(maxPeriod) {
  if (status == MILPStatus::UNSAT_PROPERTIES)
    return;

  // Compute the list of CFDFCs unions
  SmallVector<std::pair<BlockSet, SmallVector<CFDFC *>>> disjointBlockSets;
  for (auto &cfAndOpt : funcInfo.cfdfcs) {
    CFDFC *cf = cfAndOpt.first;

    auto insertFromCycle = [&](BlockSet &set) -> void {
      for (unsigned bb : cf->cycle)
        set.insert(bb);
    };

    bool foundGroup = false;
    for (auto &[blockSet, cfdfcs] : disjointBlockSets) {
      // Insert all blocks in the CFDFC cycle inside a set for comparison with
      // the blockSet
      BlockSet cycleSet;
      insertFromCycle(cycleSet);
      llvm::set_intersect(cycleSet, blockSet);

      if (cycleSet.empty())
        // Intersection between cycleSet and blockSet is empty
        continue;

      // If some element remains in cycleSet, it means that it was already in
      // blockSet and that the CFDFC belongs in that group
      insertFromCycle(blockSet);
      cfdfcs.push_back(cf);
      foundGroup = true;
      break;
    }

    if (!foundGroup) {
      // Create a new group for the CFDFC, since it has no common block with any
      // other group
      BlockSet cycleSet;
      insertFromCycle(cycleSet);
      disjointBlockSets.emplace_back(cycleSet, SmallVector<CFDFC *>{cf});
    }
  }
  for (auto &[blockSet, cfdfcs] : disjointBlockSets)
    cfdfcUnions.emplace_back(cfdfcs);

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

LogicalResult FPL22Buffers::setup() { return failure(); }

LogicalResult FPL22Buffers::createVars() {
  model.update();
  return failure();
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
