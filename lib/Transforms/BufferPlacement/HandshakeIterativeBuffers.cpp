//===- HandshakeIterativeBuffers.cpp - Iter. buffer placement ---*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// <TODO What does this file implements?>
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Transforms/BufferPlacement/HandshakeIterativeBuffers.h"
#include "dynamatic/Transforms/BufferPlacement/FPGA20Buffers.h"
#include "dynamatic/Transforms/BufferPlacement/HandshakePlaceBuffers.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"

using namespace llvm::sys;
using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace dynamatic;
using namespace dynamatic::buffer;
using namespace dynamatic::experimental;

namespace {

/// Stores sone data you may want to extract from buffer placement
struct MyData {
  unsigned someCountOfSomething = 0;
};

/// Sub-type of the classic buffer placement pass, just so that we can override
/// some of the methods used during buffer placement and extract internal data
/// (e.g., channel throughputs as determined by the MILP) from the pass.
struct MyBufferPlacementPass : public HandshakePlaceBuffersPass {
  MyBufferPlacementPass(MyData &data, StringRef algorithm,
                        StringRef frequencies, StringRef timingModels,
                        bool firstCFDFC, double targetCP, unsigned timeout,
                        bool dumpLogs)
      : HandshakePlaceBuffersPass(algorithm, frequencies, timingModels,
                                  firstCFDFC, targetCP, timeout, dumpLogs),
        data(data){};

  /// Some data you care about extracting.
  MyData &data;

protected:
  /// Custom buffer placement step. Copied from `HandshakePlaceBuffersPass` with
  /// the addition of a step at the end to extract some information from the
  /// MILP.
  LogicalResult
  getBufferPlacement(FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
                     DenseMap<Value, PlacementResult> &placement) override;
};
} // namespace

LogicalResult MyBufferPlacementPass::getBufferPlacement(
    FuncInfo &info, TimingDatabase &timingDB, Logger *logger,
    DenseMap<Value, PlacementResult> &placement) {
  /// This is exactly the same as the getBufferPlacement method in
  /// HandshakePlaceBuffersPass

  // Create Gurobi environment
  GRBEnv env = GRBEnv(true);
  env.set(GRB_IntParam_OutputFlag, 0);
  if (timeout > 0)
    env.set(GRB_DoubleParam_TimeLimit, timeout);
  env.start();

  // Create and solve the MILP
  BufferPlacementMILP *milp = nullptr;
  if (algorithm == "fpga20")
    milp = new fpga20::FPGA20Buffers(info, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, false);
  else if (algorithm == "fpga20-legacy")
    milp = new fpga20::FPGA20Buffers(info, timingDB, env, logger, targetCP,
                                     targetCP * 2.0, true);
  assert(milp && "unknown placement algorithm");
  int milpStat;
  LogicalResult res = success();
  if (failed(milp->optimize(&milpStat))) {
    res = info.funcOp->emitError()
          << "Buffer placement MILP failed with status " << milpStat
          << ", reason:" << getGurobiOptStatusDesc(milpStat);
  } else if (failed(milp->getPlacement(placement))) {
    res = info.funcOp->emitError()
          << "Failed to extract placement decisions from MILP's solution.";
  }

  // Here, before destroying the MILP, extract whatever information you want
  // and store it into your MyData& reference. If you need to extract variable
  // values from the MILP you may need to make some of its fields public (to
  // be discussed in PRs).
  llvm::errs() << "Setting some random count!\n";
  data.someCountOfSomething = 42;

  delete milp;
  return res;
}

namespace {
/// Driver for the iterative buffer placement pass, which runs buffer placement
/// and something else of your choosing iteratively until some exit condition is
/// met.
struct HandshakeIterativeBuffersPass
    : public dynamatic::buffer::impl::HandshakeIterativeBuffersBase<
          HandshakeIterativeBuffersPass> {

  /// Note that I made the pass take exactly the same set of parameters as the
  /// buffer placement pass for completeness. If some of them are useless to you
  /// (e.g., you may only want to ever use the 'fpga20' algorithm, you can
  /// remove them from there and from everywhere they are mentionned)
  HandshakeIterativeBuffersPass(StringRef algorithm, StringRef frequencies,
                                StringRef timingModels, bool firstCFDFC,
                                double targetCP, unsigned timeout,
                                bool dumpLogs) {
    this->algorithm = algorithm.str();
    this->frequencies = frequencies.str();
    this->timingModels = timingModels.str();
    this->firstCFDFC = firstCFDFC;
    this->targetCP = targetCP;
    this->timeout = timeout;
    this->dumpLogs = dumpLogs;
  }

  void runOnOperation() override;
};
} // namespace

void HandshakeIterativeBuffersPass::runOnOperation() {
  ModuleOp modOp = getOperation();

  while (true) {
    // Data object to extract information from buffer placement
    MyData data;

    // Use a pass manager to run buffer placement on the current module
    PassManager pm(&getContext());
    pm.addPass(std::make_unique<MyBufferPlacementPass>(
        data, algorithm, frequencies, timingModels, firstCFDFC, targetCP,
        timeout, dumpLogs));
    if (failed(pm.run(modOp))) {
      return signalPassFailure();
    }

    // At this point modOp is buffered. Now you can:
    // - further modify the module by applying any kind of transformation you
    //   want
    // - break out of the loop
    // - ...
    if (data.someCountOfSomething == 42) {
      llvm::errs() << "Breaking out of the loop!\n";
      break;
    }
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
dynamatic::buffer::createHandshakeIterativeBuffers(
    StringRef algorithm, StringRef frequencies, StringRef timingModels,
    bool firstCFDFC, double targetCP, unsigned timeout, bool dumpLogs) {
  return std::make_unique<HandshakeIterativeBuffersPass>(
      algorithm, frequencies, timingModels, firstCFDFC, targetCP, timeout,
      dumpLogs);
}
