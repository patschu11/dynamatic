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
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Handshake.h"
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

  MinimizeLSQUsage(NameAnalysis &namer, MLIRContext *ctx)
      : OpRewritePattern(ctx), namer(namer){};

  LogicalResult matchAndRewrite(handshake::LSQOp lsqOp,
                                PatternRewriter &rewriter) const override;

private:
  NameAnalysis &namer;

  struct LSQInfo {
    /// Maps LSQ load and store ports to the index of the group they belong to.
    DenseMap<Operation *, unsigned> lsqPortToGroup;
    /// All loads to the LSQ, in group order.
    SetVector<handshake::LSQLoadOp> loadOps;
    /// All stores to the LSQ, in group order.
    SetVector<handshake::LSQStoreOp> storeOps;
    /// Names of loads to the LSQ that may go directly to an MC.
    DenseSet<StringRef> removableLoads;
    /// Names of stores to the LSQ that may go directly to an MC.
    DenseSet<StringRef> removableStores;
    /// Maps basic block IDs to their control value, for reconnecting memory
    /// interfaces to the circuit in case the LSQ is optimizable.
    DenseMap<unsigned, Value> ctrlVals;
    /// Whether the LSQ is optimizable.
    bool optimizable = false;

    LSQInfo(handshake::LSQOp lsqOp);
  };

  void replaceEndMemoryControls(handshake::LSQOp lsqOp,
                                handshake::LSQOp newLSQOp,
                                handshake::MemoryControllerOp newMCOp,
                                PatternRewriter &rewriter) const;
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

    // Check that all eligible operations within Handshake function belon to a
    // basic block
    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      for (Operation &op : funcOp.getOps()) {
        if (!getLogicBB(&op)) {
          op.emitError() << "Operation should have basic block "
                            "attribute.";
          return signalPassFailure();
        }
      }
    }

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    RewritePatternSet patterns{ctx};
    patterns.add<MinimizeLSQUsage>(namer, ctx);
    if (failed(
            applyPatternsAndFoldGreedily(modOp, std::move(patterns), config)))
      return signalPassFailure();
  }
};
} // namespace

static bool hasRAW(handshake::LSQLoadOp loadOp,
                   SetVector<handshake::LSQStoreOp> &storeOps) {
  StringRef loadName = getUniqueName(loadOp);
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
                              handshake::LSQStoreOp storeOp,
                              HandshakeCFG &cfg) {
  // Identify all CFG paths from the block containing the load to the block
  // containing the store
  handshake::FuncOp funcOp = loadOp->getParentOfType<handshake::FuncOp>();
  assert(funcOp && "parent of load access must be handshake function");
  SmallVector<CFGPath> allPaths;
  std::optional<unsigned> loadBB = getLogicBB(loadOp);
  std::optional<unsigned> storeBB = getLogicBB(storeOp);
  assert(loadBB && storeBB && "memory accesses must belong to blocks");
  cfg.getNonCyclicPaths(*loadBB, *storeBB, allPaths);

  // There must be a dependence between any operand of the store with the load
  // data result on all CFG paths between them
  Value loadData = loadOp.getDataResult();
  return llvm::all_of(allPaths, [&](CFGPath &path) {
    return isGIID(loadData, storeOp.getDataInput(), path) ||
           isGIID(loadData, storeOp.getAddressResult(), path);
  });
}

static bool hasEnforcedWARs(handshake::LSQLoadOp loadOp,
                            SetVector<handshake::LSQStoreOp> &storeOps,
                            HandshakeCFG &cfg) {
  DenseMap<StringRef, handshake::LSQStoreOp> storesByName;
  for (handshake::LSQStoreOp storeOp : storeOps)
    storesByName[getUniqueName(storeOp)] = storeOp;

  // We only need to check stores that depend on the load (WAR dependencies) as
  // others are already provably independent. We may check a single store
  // multiple times if it depends on the load at multiple loop depths
  auto deps = getUniqueAttr<MemDependenceArrayAttr>(loadOp);
  for (MemDependenceAttr dependency : deps.getDependencies()) {
    handshake::LSQStoreOp storeOp = storesByName[dependency.getDstAccess()];
    if (!isStoreGIIDOnLoad(loadOp, storeOp, cfg))
      return false;
  }
  return true;
}

static bool isStoreRemovable(handshake::LSQStoreOp storeOp,
                             DenseSet<StringRef> &independentAccesses) {
  auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp);
  return llvm::all_of(
      deps.getDependencies(), [&](MemDependenceAttr dependency) {
        return independentAccesses.find(dependency.getDstAccess().str()) !=
               independentAccesses.end();
      });
}

LogicalResult
MinimizeLSQUsage::matchAndRewrite(handshake::LSQOp lsqOp,
                                  PatternRewriter &rewriter) const {
  // Check whether the LSQ is optimizable
  LSQInfo lsqInfo(lsqOp);
  if (!lsqInfo.optimizable)
    return failure();

  // Context for creating new operation
  MLIRContext *ctx = getContext();

  // Used to keep track of memory replacements so that dependencies between
  // accesses stay consistent
  MemoryOpLowering memOpLowering(namer);

  // Used to instantiate new memory interfaces after transforming some of our
  // LSQ ports into MC ports
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  Value memref = lsqOp.getMemRef();
  MemoryInterfaceBuilder memBuilder(funcOp, memref, lsqInfo.ctrlVals);

  // Replace removable LSQ loads with their MC variant
  for (handshake::LSQLoadOp lsqLoadOp : lsqInfo.loadOps) {
    if (lsqInfo.removableLoads.contains(getUniqueName(lsqLoadOp))) {
      rewriter.setInsertionPoint(lsqLoadOp);
      auto mcLoadOp = rewriter.create<handshake::MCLoadOp>(
          lsqLoadOp->getLoc(), lsqLoadOp.getDataInput().getType(),
          lsqLoadOp.getAddressInput());
      inheritBB(lsqLoadOp, mcLoadOp);

      // Replace operation's data result which goes to the circuit
      rewriter.replaceAllUsesWith(lsqLoadOp.getDataResult(),
                                  mcLoadOp.getDataResult());

      // Record operation replacement (change interface to MC)
      memOpLowering.recordReplacement(lsqLoadOp, mcLoadOp, false);
      setUniqueAttr(mcLoadOp, handshake::MemInterfaceAttr::get(ctx));
      rewriter.eraseOp(lsqLoadOp);

      memBuilder.addMCPort(*getLogicBB(mcLoadOp), mcLoadOp);
    } else {
      memBuilder.addLSQPort(lsqInfo.lsqPortToGroup[lsqLoadOp], lsqLoadOp);
    }
  }

  // Replace removable LSQ stores with their MC variant
  for (handshake::LSQStoreOp lsqStoreOp : lsqInfo.storeOps) {
    if (lsqInfo.removableStores.contains(getUniqueName(lsqStoreOp))) {
      rewriter.setInsertionPoint(lsqStoreOp);
      auto mcStoreOp = rewriter.create<handshake::MCStoreOp>(
          lsqStoreOp->getLoc(), lsqStoreOp.getAddressInput(),
          lsqStoreOp.getDataInput());
      inheritBB(lsqStoreOp, mcStoreOp);

      // Record operation replacement (change interface to MC)
      memOpLowering.recordReplacement(lsqStoreOp, mcStoreOp, false);
      setUniqueAttr(mcStoreOp, handshake::MemInterfaceAttr::get(ctx));
      rewriter.eraseOp(lsqStoreOp);

      memBuilder.addMCPort(*getLogicBB(mcStoreOp), mcStoreOp);
    } else {
      memBuilder.addLSQPort(lsqInfo.lsqPortToGroup[lsqStoreOp], lsqStoreOp);
    }
  }

  // Rename memory accesses referenced by memory dependencies attached to the
  // old and new memory ports
  memOpLowering.renameDependencies(lsqOp->getParentOp());

  // Instantiate new memory interfaces
  handshake::MemoryControllerOp newMCOp;
  handshake::LSQOp newLSQOp;
  if (failed(memBuilder.instantiateInterfaces(rewriter, newMCOp, newLSQOp)))
    return failure();

  // Replace memory control signals consumed by the end operation
  replaceEndMemoryControls(lsqOp, newLSQOp, newMCOp, rewriter);

  // Erase the original LSQ and potential MC to the same memory interface
  if (handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC())
    rewriter.eraseOp(mcOp);
  rewriter.eraseOp(lsqOp);
  return success();
}

MinimizeLSQUsage::LSQInfo::LSQInfo(handshake::LSQOp lsqOp) {
  for (auto [idx, group] : llvm::enumerate(lsqOp.getPorts().getGroups())) {
    for (MemoryPort &port : group->accessPorts) {
      if (std::optional<LSQLoadPort> loadPort = dyn_cast<LSQLoadPort>(port)) {
        handshake::LSQLoadOp lsqLoadOp = loadPort->getLSQLoadOp();
        lsqPortToGroup[lsqLoadOp] = idx;
        loadOps.insert(lsqLoadOp);
      } else if (std::optional<LSQStorePort> storePort =
                     dyn_cast<LSQStorePort>(port)) {
        handshake::LSQStoreOp lsqStoreOp = storePort->getLSQStoreOp();
        lsqPortToGroup[lsqStoreOp] = idx;
        storeOps.insert(storePort->getLSQStoreOp());
      }
    }
  }

  // We will need CFG information about the containing Handshake function
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  HandshakeCFG cfg(funcOp);

  // Compute the set of loads that can go directly to an MC inside of an LSQ
  DenseSet<StringRef> removableLoads;
  for (handshake::LSQLoadOp loadOp : loadOps) {
    // Loads with no RAW dependencies and which satisfy the GIID property with
    // all stores may be removed
    if (!hasRAW(loadOp, storeOps) && hasEnforcedWARs(loadOp, storeOps, cfg))
      removableLoads.insert(getUniqueName(loadOp));
  }
  if (removableLoads.empty())
    return;

  // Compute the set of stores that can go directly to an MC inside of an LSQ
  // now that we know that some loads are out
  DenseSet<StringRef> removableStores;
  for (handshake::LSQStoreOp storeOp : storeOps) {
    if (isStoreRemovable(storeOp, removableLoads))
      removableStores.insert(getUniqueName(storeOp));
  }

  // We need the control value of each block in the Handshake function to be
  // able to recreate memory interfaces
  if (failed(cfg.getControlValues(ctrlVals)))
    return;

  optimizable = true;
}

void MinimizeLSQUsage::replaceEndMemoryControls(
    handshake::LSQOp lsqOp, handshake::LSQOp newLSQOp,
    handshake::MemoryControllerOp newMCOp, PatternRewriter &rewriter) const {
  // We must update control signals going out of the memory interfaces and to
  // the function's terminator. The number of total memory interfaces may have
  // changed (a new MC may have appeared, an LSQ may have disappeared) and
  // consequently the number of operands to the function's terminator may change
  // too
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  handshake::EndOp endOp = dyn_cast<EndOp>(funcOp.front().getTerminator());
  assert(endOp && "expected end operation in Handshake function");
  handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC();

  // Derive new memory control operands for the end operation
  SmallVector<Value> newEndOperands(endOp.getReturnValues());
  bool needNewMCControl = newMCOp != nullptr;
  for (Value memCtrl : endOp.getMemoryControls()) {
    if (mcOp && memCtrl == mcOp.getDone()) {
      // In this case we are guaranteed to have instantiated a new MC, as we
      // never modify MC ports as part of the optimization
      newEndOperands.push_back(newMCOp.getDone());
      needNewMCControl = false;
    } else if (memCtrl == lsqOp.getDone()) {
      // Skip the control signal if we did not reinstantiate an LSQ due to the
      // optimization
      if (newLSQOp)
        newEndOperands.push_back(newLSQOp.getDone());
    } else {
      newEndOperands.push_back(memCtrl);
    }
  }
  // If we have inserted a MC that was not present before, then its done signal
  // need to be added to the list of memory control signals
  if (needNewMCControl)
    newEndOperands.push_back(newMCOp.getDone());

  // Replace the end operation
  rewriter.setInsertionPoint(endOp);
  auto newEndOp =
      rewriter.create<handshake::EndOp>(endOp.getLoc(), newEndOperands);
  inheritBB(endOp, newEndOp);
  rewriter.eraseOp(endOp);
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeLSQUsage() {
  return std::make_unique<HandshakeMinimizeLSQUsagePass>();
}
