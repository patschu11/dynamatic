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
#include "circt/Support/BackedgeBuilder.h"
#include "dynamatic/Analysis/NameAnalysis.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "dynamatic/Support/Handshake.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

namespace {

/// TODO
struct OptimizeLSQ : public OpRewritePattern<handshake::LSQOp> {
  using OpRewritePattern<handshake::LSQOp>::OpRewritePattern;

  OptimizeLSQ(NameAnalysis &namer, MLIRContext *ctx)
      : OpRewritePattern(ctx), namer(namer){};

  LogicalResult matchAndRewrite(handshake::LSQOp lsqOp,
                                PatternRewriter &rewriter) const override;

private:
  /// Reference to the calling pass's naming analysis, used to name new memory
  /// operations on the fly as they are being created.
  NameAnalysis &namer;

  struct LSQInfo {
    /// Maps LSQ load and store ports to the index of the group they belong to.
    DenseMap<Operation *, unsigned> lsqPortToGroup;
    /// All loads to the LSQ, in group order.
    SetVector<handshake::LSQLoadOp> lsqLoadOps;
    /// All stores to the LSQ, in group order.
    SetVector<handshake::LSQStoreOp> lsqStoreOps;
    /// All accesses to a potential MC connected to the LSQ, in block order.
    SetVector<Operation *> mcOps;
    /// Names of loads to the LSQ that may go directly to an MC.
    DenseSet<StringRef> removableLoads;
    /// Names of stores to the LSQ that may go directly to an MC.
    DenseSet<StringRef> removableStores;
    /// Maps basic block IDs to their control value, for reconnecting memory
    /// interfaces to the circuit in case the LSQ is optimizable.
    DenseMap<unsigned, Value> ctrlVals;
    /// Whether the LSQ is optimizable.
    bool optimizable = false;

    /// Determines whether the LSQ is optimizable, filling in all struct members
    /// in the process. First, stores the list of loads and stores to the LSQ,
    /// then analyses the DFG to potentially identify accesses that do not need
    /// to go through an LSQ because of control-flow-enforced dependencies, and
    /// finally determines whether the LSQ is optimizable.
    LSQInfo(handshake::LSQOp lsqOp);
  };

  /// Updates the parent function's terminator's operands to reflect the changes
  /// in memory interfaces, which all produce a done signal consumed by the
  /// terminator. New memory interfaces may be nullptr.
  void replaceEndMemoryControls(handshake::LSQOp lsqOp,
                                handshake::LSQOp newLSQOp,
                                handshake::MemoryControllerOp newMCOp,
                                PatternRewriter &rewriter) const;
};

/// TODO
struct HandshakeMinimizeLSQUsagePass
    : public dynamatic::impl::HandshakeMiminizeLSQUsageBase<
          HandshakeMinimizeLSQUsagePass> {

  void runDynamaticPass() override;
};
} // namespace

static bool hasRAW(handshake::LSQLoadOp loadOp,
                   SetVector<handshake::LSQStoreOp> &storeOps) {
  StringRef loadName = getUniqueName(loadOp);
  for (handshake::LSQStoreOp storeOp : storeOps) {
    if (auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp)) {
      for (MemDependenceAttr dependency : deps.getDependencies()) {
        if (dependency.getDstAccess() == loadName)
          return true;
      }
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
    return isGIID(loadData, storeOp.getDataInput(), storeOp, path) ||
           isGIID(loadData, storeOp.getAddressResult(), storeOp, path);
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
  if (auto deps = getUniqueAttr<MemDependenceArrayAttr>(loadOp)) {
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      handshake::LSQStoreOp storeOp = storesByName[dependency.getDstAccess()];
      assert(storeOp && "unknown store operation");
      if (!isStoreGIIDOnLoad(loadOp, storeOp, cfg))
        return false;
    }
  }
  return true;
}

static bool isStoreRemovable(handshake::LSQStoreOp storeOp,
                             DenseSet<StringRef> &independentAccesses) {
  auto deps = getUniqueAttr<MemDependenceArrayAttr>(storeOp);
  if (!deps || deps.getDependencies().empty())
    return true;
  return llvm::all_of(
      deps.getDependencies(), [&](MemDependenceAttr dependency) {
        return independentAccesses.find(dependency.getDstAccess().str()) !=
               independentAccesses.end();
      });
}

LogicalResult OptimizeLSQ::matchAndRewrite(handshake::LSQOp lsqOp,
                                           PatternRewriter &rewriter) const {
  llvm::errs() << "Calling opt pattern\n";
  // Check whether the LSQ is optimizable
  LSQInfo lsqInfo(lsqOp);
  if (!lsqInfo.optimizable)
    return failure();

  llvm::errs() << "Removing " << lsqInfo.removableLoads.size() << " loads and "
               << lsqInfo.removableStores.size() << " stores\n";

  // Context for creating new operation
  MLIRContext *ctx = getContext();

  // Used to keep track of memory replacements so that dependencies between
  // accesses stay consistent
  MemoryOpLowering memOpLowering(namer);

  // Used to instantiate new memory interfaces after transforming some of our
  // LSQ ports into MC ports
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  Value memref = lsqOp.getMemRef();
  MemRefType memType = cast<MemRefType>(memref.getType());
  MemoryInterfaceBuilder memBuilder(funcOp, memref, lsqInfo.ctrlVals);

  // Existing memory ports and memory interface(s) reference each other's
  // results/operands, which makes them un-earasable since it's disallowed to
  // remove an operation whose results still have active uses. Use temporary
  // backedges to replace the to-be-removed memory ports' results in the memory
  // interface(s) operands, which allows us to first delete the memory ports and
  // finally the memory interfaces. All backedges are deleted automatically
  // before the method retuns
  BackedgeBuilder backedgeBuilder(rewriter, lsqOp.getLoc());

  // Collect all memory accesses that must be rerouted to new memory interfaces.
  // It's important to iterate in operation order here to maintain the original
  // program order in each memory group
  for (Operation &op : llvm::make_early_inc_range(funcOp.getOps())) {
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<handshake::LSQLoadOp>([&](handshake::LSQLoadOp lsqLoadOp) {
          if (!lsqInfo.removableLoads.contains(getUniqueName(lsqLoadOp))) {
            memBuilder.addLSQPort(lsqInfo.lsqPortToGroup[lsqLoadOp], lsqLoadOp);
            return;
          }

          // Replace the LSQ load with an equivalent MC load
          rewriter.setInsertionPoint(lsqLoadOp);
          auto mcLoadOp = rewriter.create<handshake::MCLoadOp>(
              lsqLoadOp->getLoc(), memType, lsqLoadOp.getAddressInput());
          inheritBB(lsqLoadOp, mcLoadOp);

          // Record operation replacement (change interface to MC)
          memOpLowering.recordReplacement(lsqLoadOp, mcLoadOp, false);
          setUniqueAttr(mcLoadOp, handshake::MemInterfaceAttr::get(ctx));
          memBuilder.addMCPort(mcLoadOp);

          // Replace the original port operation's results and erase it
          rewriter.replaceAllUsesWith(lsqLoadOp.getDataResult(),
                                      mcLoadOp.getDataResult());
          Value addrOut = lsqLoadOp.getAddressOutput();
          rewriter.replaceAllUsesWith(addrOut,
                                      backedgeBuilder.get(addrOut.getType()));
          rewriter.eraseOp(lsqLoadOp);
        })
        .Case<handshake::LSQStoreOp>([&](handshake::LSQStoreOp lsqStoreOp) {
          if (!lsqInfo.removableStores.contains(getUniqueName(lsqStoreOp))) {
            memBuilder.addLSQPort(lsqInfo.lsqPortToGroup[lsqStoreOp],
                                  lsqStoreOp);
            return;
          }

          // Replace the LSQ store with an equivalent MC store
          rewriter.setInsertionPoint(lsqStoreOp);
          auto mcStoreOp = rewriter.create<handshake::MCStoreOp>(
              lsqStoreOp->getLoc(), lsqStoreOp.getAddressInput(),
              lsqStoreOp.getDataInput());
          inheritBB(lsqStoreOp, mcStoreOp);

          // Record operation replacement (change interface to MC)
          memOpLowering.recordReplacement(lsqStoreOp, mcStoreOp, false);
          setUniqueAttr(mcStoreOp, handshake::MemInterfaceAttr::get(ctx));
          memBuilder.addMCPort(mcStoreOp);

          // Replace the original port operation's results and erase it
          Value addrOut = lsqStoreOp.getAddressOutput();
          rewriter.replaceAllUsesWith(addrOut,
                                      backedgeBuilder.get(addrOut.getType()));
          Value dataOut = lsqStoreOp.getDataOutput();
          rewriter.replaceAllUsesWith(dataOut,
                                      backedgeBuilder.get(dataOut.getType()));
          rewriter.eraseOp(lsqStoreOp);
        })
        .Case<handshake::MCLoadOp>([&](handshake::MCLoadOp mcLoadOp) {
          // The data operand coming from the current memory interface will be
          // replaced during interface creation by the `MemoryInterfaceBuilder`
          if (lsqInfo.mcOps.contains(mcLoadOp))
            memBuilder.addMCPort(mcLoadOp);
        })
        .Case<handshake::MCStoreOp>([&](handshake::MCStoreOp mcStoreOp) {
          if (lsqInfo.mcOps.contains(mcStoreOp))
            memBuilder.addMCPort(mcStoreOp);
        });
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

  // If the LSQ is connected to an MC, we delete it first. The second to last
  // result of the MC is a load data signal going to the LSQ, which needs to be
  // temporarily replaced with a backedge to allow us to remove the MC before
  // the LSQ
  if (handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC()) {
    rewriter.setInsertionPoint(mcOp);
    Value loadDataToLSQ = mcOp.getResult(mcOp.getNumResults() - 2);
    rewriter.replaceAllUsesWith(loadDataToLSQ,
                                backedgeBuilder.get(loadDataToLSQ.getType()));

    for (OpResult res : mcOp.getResults()) {
      llvm::errs() << "MC res has "
                   << std::distance(res.getUses().begin(), res.getUses().end())
                   << " uses\n";
      for (Operation *user : res.getUsers())
        user->emitRemark();
    }
    rewriter.eraseOp(mcOp);
  }
  for (OpResult res : lsqOp.getResults())
    llvm::errs() << "LSQ res has "
                 << std::distance(res.getUses().begin(), res.getUses().end())
                 << " uses\n";

  rewriter.eraseOp(lsqOp);
  llvm::errs() << "Optimized!\n";
  return success();
}

OptimizeLSQ::LSQInfo::LSQInfo(handshake::LSQOp lsqOp) {
  // Identify load and store accesses to the LSQ
  LSQPorts ports = lsqOp.getPorts();
  for (auto [idx, group] : llvm::enumerate(ports.getGroups())) {
    for (MemoryPort &port : group->accessPorts) {
      if (std::optional<LSQLoadPort> loadPort = dyn_cast<LSQLoadPort>(port)) {
        handshake::LSQLoadOp lsqLoadOp = loadPort->getLSQLoadOp();
        lsqPortToGroup[lsqLoadOp] = idx;
        lsqLoadOps.insert(lsqLoadOp);
      } else if (std::optional<LSQStorePort> storePort =
                     dyn_cast<LSQStorePort>(port)) {
        handshake::LSQStoreOp lsqStoreOp = storePort->getLSQStoreOp();
        lsqPortToGroup[lsqStoreOp] = idx;
        lsqStoreOps.insert(lsqStoreOp);
      }
    }
  }

  // We will need CFG information about the containing Handshake function
  handshake::FuncOp funcOp = lsqOp->getParentOfType<handshake::FuncOp>();
  HandshakeCFG cfg(funcOp);

  // Compute the set of loads that can go directly to an MC inside of an LSQ
  for (handshake::LSQLoadOp loadOp : lsqLoadOps) {
    // Loads with no RAW dependencies and which satisfy the GIID property with
    // all stores may be removed
    if (!hasRAW(loadOp, lsqStoreOps) &&
        hasEnforcedWARs(loadOp, lsqStoreOps, cfg))
      removableLoads.insert(getUniqueName(loadOp));
  }
  if (removableLoads.empty())
    return;

  // Compute the set of stores that can go directly to an MC inside of an LSQ
  // now that we know that some loads are out
  for (handshake::LSQStoreOp storeOp : lsqStoreOps) {
    if (isStoreRemovable(storeOp, removableLoads))
      removableStores.insert(getUniqueName(storeOp));
  }

  // We need the control value of each block in the Handshake function to be
  // able to recreate memory interfaces
  if (failed(cfg.getControlValues(ctrlVals)))
    return;

  // If the LSQ connects to an MC, memory accesses going directly to the MC will
  // also need to be rerouted
  if (handshake::MemoryControllerOp mcOp = lsqOp.getConnectedMC()) {
    MCPorts mcPorts = mcOp.getPorts();
    for (auto [idx, group] : llvm::enumerate(mcPorts.getBlocks())) {
      for (MemoryPort &port : group->accessPorts) {
        if (isa<MCLoadPort, MCStorePort>(port))
          mcOps.insert(port.portOp);
      }
    }
  }

  optimizable = true;
}

void OptimizeLSQ::replaceEndMemoryControls(
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
  SmallVector<Value> newEndOperands;
  bool needNewMCControl = newMCOp != nullptr;
  for (Value memCtrl : endOp.getOperands()) {
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

void HandshakeMinimizeLSQUsagePass::runDynamaticPass() {
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
      if (!cannotBelongToCFG(&op) && !getLogicBB(&op)) {
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
  patterns.add<OptimizeLSQ>(namer, ctx);
  if (failed(applyPatternsAndFoldGreedily(modOp, std::move(patterns), config)))
    return signalPassFailure();
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeLSQUsage() {
  return std::make_unique<HandshakeMinimizeLSQUsagePass>();
}
