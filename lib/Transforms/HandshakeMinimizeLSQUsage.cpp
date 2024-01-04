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

static Value getMCControlVal(handshake::MemoryControllerOp mcOp, MCPorts &ports,
                             unsigned bb) {
  // Find the basic block
}

static Value getLSQControlVal(handshake::LSQOp lsqOp, LSQPorts &ports,
                              unsigned groupIdx) {
  unsigned ctrlIdx = ports.getGroup(groupIdx)->ctrlPort->getCtrlInputIndex();
  Value ctrlVal = lsqOp.getMemOperands()[ctrlIdx];
  Operation *defOp = ctrlVal.getDefiningOp();
  while (isa_and_present<handshake::ForkOp, handshake::LSQOp>(defOp)) {
    ctrlVal = defOp->getOperand(0);
    defOp = ctrlVal.getDefiningOp();
  }
  return ctrlVal;
}

LogicalResult
MinimizeLSQUsage::matchAndRewrite(handshake::LSQOp lsqOp,
                                  PatternRewriter &rewriter) const {

  // Maps LSQ load and store ports to the index of the group they belong to
  DenseMap<Operation *, unsigned> lsqPortToGroup;

  // Collect loads and stores to the LSQ
  SetVector<handshake::LSQLoadOp> loadOps;
  SetVector<handshake::LSQStoreOp> storeOps;
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
  HandshakeCFG cfg(lsqOp->getParentOfType<handshake::FuncOp>());

  // Compute the set of loads that can go directly to an MC inside of an LSQ
  DenseSet<StringRef> removableLoads;
  for (handshake::LSQLoadOp loadOp : loadOps) {
    // Check for RAW dependencies with the load and for the GIID property
    // between all the stores and the load
    if (!hasRAW(loadOp, storeOps) && hasEnforcedWARs(loadOp, storeOps, cfg))
      removableLoads.insert(getUniqueName(loadOp));
  }
  if (removableLoads.empty())
    return failure();

  // Compute the set of stores that can go directly to an MC inside of an LSQ
  // now that we know that some loads are out
  DenseSet<StringRef> removableStores;
  for (handshake::LSQStoreOp storeOp : storeOps) {
    if (isStoreRemovable(storeOp, removableLoads))
      removableStores.insert(getUniqueName(storeOp));
  }

  // Used to keep track of memory replacements so that dependencies between
  // accesses stay consistent
  MemoryOpLowering memOpLowering(namer);

  // Context for creating new operation
  MLIRContext *ctx = getContext();

  // We need the control value of each block in the Handshake function to be
  // able to recreate memory interfaces
  DenseMap<unsigned, Value> ctrlVals;
  if (failed(cfg.getControlValues(ctrlVals)))
    return failure();

  // Group ports of a future MC by basic block
  llvm::MapVector<unsigned, SmallVector<Operation *>> mcPorts;
  // Group ports of a future LSQ by group
  llvm::MapVector<unsigned, SmallVector<Operation *>> lsqPorts;
  // Count numver of loads to the future MC and LSQ
  unsigned mcNumLoads = 0, lsqNumLoads = 0;

  // Replace removable LSQ loads with their MC variant
  for (handshake::LSQLoadOp lsqLoadOp : loadOps) {
    if (removableLoads.contains(getUniqueName(lsqLoadOp))) {
      rewriter.setInsertionPoint(lsqLoadOp);
      auto mcLoadOp = rewriter.create<handshake::MCLoadOp>(
          lsqLoadOp->getLoc(), lsqLoadOp.getDataInput().getType(),
          lsqLoadOp.getAddressInput());
      inheritBB(lsqLoadOp, mcLoadOp);

      // Record operation replacement
      memOpLowering.recordReplacement(lsqLoadOp, mcLoadOp, false);
      setUniqueAttr(mcLoadOp, handshake::MemInterfaceAttr::get(ctx));

      // Replace data result (address result will be replaced later, as it goes
      // to a memory inteface thay is not yet created)
      rewriter.replaceAllUsesWith(lsqLoadOp.getDataResult(),
                                  mcLoadOp.getDataResult());

      ++mcNumLoads;
      mcPorts[*getLogicBB(mcLoadOp)].push_back(mcLoadOp);
    } else {
      ++lsqNumLoads;
      lsqPorts[lsqPortToGroup[lsqLoadOp]].push_back(lsqLoadOp);
    }
  }

  // Replace removable LSQ stores with their MC variant
  for (handshake::LSQStoreOp lsqStoreOp : storeOps) {
    if (removableStores.contains(getUniqueName(lsqStoreOp))) {
      rewriter.setInsertionPoint(lsqStoreOp);
      auto mcStoreOp = rewriter.create<handshake::MCStoreOp>(
          lsqStoreOp->getLoc(), lsqStoreOp.getAddressInput(),
          lsqStoreOp.getDataInput());
      inheritBB(lsqStoreOp, mcStoreOp);

      // Record operation replacement (change interface to MC)
      memOpLowering.recordReplacement(lsqStoreOp, mcStoreOp, false);
      setUniqueAttr(mcStoreOp, handshake::MemInterfaceAttr::get(ctx));

      mcPorts[*getLogicBB(mcStoreOp)].push_back(mcStoreOp);
    } else {
      lsqPorts[lsqPortToGroup[lsqStoreOp]].push_back(lsqStoreOp);
    }
  }

  // Rename memory accesses referenced by memory dependencies attached to the
  // old and new memory ports
  memOpLowering.renameDependencies(lsqOp->getParentOp());

  // Retrieve MC attached to the LSQ (may be nullptr)
  handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC();

  return success();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeLSQUsage() {
  return std::make_unique<HandshakeMinimizeLSQUsagePass>();
}
