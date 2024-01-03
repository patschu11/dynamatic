//===- HandshakeMiminizeLSQUsage.cpp - LSQ flow analysis --------*- C++ -*-===//
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

#include "dynamatic/Transforms/HandshakeMinimizeLSQUsage.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/Handshake.h"
#include "dynamatic/Support/LogicBB.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

namespace {

/// TODO
struct MinimizeLSQUsage : public OpRewritePattern<handshake::LSQOp> {
  using OpRewritePattern<handshake::LSQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::LSQOp lsqOp,
                                PatternRewriter &rewriter) const override;
};

/// TODO
struct HandshakeMinimizeLSQUsagePass
    : public dynamatic::impl::HandshakeMiminizeLSQUsageBase<
          HandshakeMinimizeLSQUsagePass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    MLIRContext *ctx = &getContext();

    // Check that memory access ports are named
    NameAnalysis &namer = getAnalysis<NameAnalysis>();
    WalkResult res = modOp.walk([&](Operation *op) {
      if (!isa<handshake::LoadOpInterface, handshake::StoreOpInterface>(op))
        return WalkResult::advance();
      if (!namer.hasName(op)) {
        op->emitError() << "Memory access port must be named.";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      return signalPassFailure();

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<MinimizeLSQUsage>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(modOp, std::move(patterns), config)))
      return signalPassFailure();
  }
};
} // namespace

static bool hasRAW(handshake::LSQLoadOp loadOp,
                   SetVector<handshake::LSQStoreOp> &storeOps) {
  std::string loadName = getUniqueName(loadOp);
  for (handshake::LSQStoreOp storeOp : storeOps) {
    auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp);
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      if (dependency.getDstAccess() == loadName)
        return true;
    }
  }
  return false;
}

static bool isStoreGIIDOnLoad(handshake::LSQLoadOp loadOp,
                              handshake::LSQStoreOp storeOp) {

  return false;
}

static bool hasEnforcedWARs(handshake::LSQLoadOp loadOp,
                            SetVector<handshake::LSQStoreOp> &storeOps) {
  DenseMap<StringRef, handshake::LSQStoreOp> storesByName;
  for (handshake::LSQStoreOp storeOp : storeOps)
    storesByName[getUniqueName(storeOp)] = storeOp;

  // We only need to check stores that depend on the load (WAR dependencies) as
  // others are already provably independent. We may check a single store
  // multiple times if it depends on the load at multiple loop depths
  auto deps = getUniqueAttr<MemDependenceArrayAttr>(loadOp);
  for (MemDependenceAttr dependency : deps.getDependencies()) {
    handshake::LSQStoreOp storeOp = storesByName[dependency.getDstAccess()];
    if (!isStoreGIIDOnLoad(loadOp, storeOp))
      return false;
  }
  return true;
}

static bool isStoreRemovable(handshake::LSQStoreOp storeOp,
                             SetVector<StringRef> &independentAccesses) {
  auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp);
  return llvm::all_of(
      deps.getDependencies(), [&](MemDependenceAttr dependency) {
        return independentAccesses.contains(dependency.getDstAccess());
      });
}

LogicalResult
MinimizeLSQUsage::matchAndRewrite(handshake::LSQOp lsqOp,
                                  PatternRewriter &rewriter) const {

  // Collect loads and stores to the LSQ
  SetVector<handshake::LSQLoadOp> loadOps;
  SetVector<handshake::LSQStoreOp> storeOps;
  for (LSQGroup &group : lsqOp.getPorts().getGroups()) {
    for (MemoryPort &port : group->accessPorts) {
      if (std::optional<LSQLoadPort> loadPort = dyn_cast<LSQLoadPort>(port)) {
        loadOps.insert(loadPort->getLSQLoadOp());
      } else if (std::optional<LSQStorePort> storePort =
                     dyn_cast<LSQStorePort>(port)) {
        storeOps.insert(storePort->getLSQStoreOp());
      }
    }
  }

  // Compute the set of loads that can be removed from the LSQ
  SetVector<StringRef> removableLoads;
  for (handshake::LSQLoadOp loadOp : loadOps) {
    // Check for RAW dependencies with the load and for the GIID property
    // between all the stores and the load
    if (!hasRAW(loadOp, storeOps) && hasEnforcedWARs(loadOp, storeOps))
      removableLoads.insert(getUniqueName(loadOp));
  }
  if (removableLoads.empty())
    return failure();

  // Compute the set of stores that can be removed from the LSQ now we know that
  // some loads are out
  SetVector<StringRef> removableStores;
  for (handshake::LSQStoreOp storeOp : storeOps) {
    if (isStoreRemovable(storeOp, removableLoads))
      removableStores.insert(getUniqueName(storeOp));
  }

  return success();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeLSQUsage() {
  return std::make_unique<HandshakeMinimizeLSQUsagePass>();
}
