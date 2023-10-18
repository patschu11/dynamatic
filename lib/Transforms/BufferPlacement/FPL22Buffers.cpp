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
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "dynamatic/Support/LogicBB.h"
#include "dynamatic/Transforms/BufferPlacement/BufferingProperties.h"
#include "dynamatic/Transforms/BufferPlacement/CFDFC.h"
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
#include "llvm/ADT/SetOperations.h"

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

LogicalResult FPL22Buffers::setup() { return failure(); }

LogicalResult FPL22Buffers::createVars() {
  model.update();
  return failure();
}

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
