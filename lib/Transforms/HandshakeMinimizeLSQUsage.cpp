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
#include "dynamatic/Support/Handshake.h"
#include "dynamatic/Support/LogicBB.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

static bool rawToLoadExists(handshake::LSQLoadOp loadOp,
                            SetVector<handshake::LSQStoreOp> &storeOps,
                            NameAnalysis &nameAnalysis) {
  StringRef loadName = nameAnalysis.getName(loadOp);
  for (handshake::LSQStoreOp storeOp : storeOps) {
    StringRef depsName = MemDependenceArrayAttr::getMnemonic();
    auto deps = storeOp->getAttrOfType<MemDependenceArrayAttr>(depsName);
    for (MemDependenceAttr dependency : deps.getDependencies()) {
      if (dependency.getDstAccess() == loadName)
        return true;
    }
  }
  return false;
}

/// Checks whether all store operations are globally in-order dependent on the
/// load operation.
static bool storesAreGIIDOnLoad(handshake::LSQLoadOp loadOp,
                                SetVector<handshake::LSQStoreOp> &storeOps) {
  Value ldData = loadOp.getDataResult();
  return llvm::all_of(storeOps, [&](handshake::LSQStoreOp storeOp) {
    return isGIID(ldData, storeOp.getDataInput()) ||
           isGIID(ldData, storeOp.getAddressInput());
  });
}

namespace {

/// TODO
struct HandshakeMinimizeLSQUsagePass
    : public dynamatic::impl::HandshakeMiminizeLSQUsageBase<
          HandshakeMinimizeLSQUsagePass> {

  void runDynamaticPass() override {
    mlir::ModuleOp modOp = getOperation();
    OpBuilder builder(&getContext());

    for (handshake::FuncOp funcOp : modOp.getOps<handshake::FuncOp>()) {
      for (handshake::LSQOp lsqOp :
           llvm::make_early_inc_range(funcOp.getOps<handshake::LSQOp>())) {
        tryToReduceLSQUsage(lsqOp, builder);
      }
    }
  }

  void tryToReduceLSQUsage(handshake::LSQOp lsqOp, OpBuilder &builder);
};
} // namespace

void HandshakeMinimizeLSQUsagePass::tryToReduceLSQUsage(handshake::LSQOp lsqOp,
                                                        OpBuilder &builder) {
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
  SetVector<handshake::LSQLoadOp> removableLoads;
  NameAnalysis &nameAnalysis = getAnalysis<NameAnalysis>();
  for (handshake::LSQLoadOp loadOp : loadOps) {
    // Check for RAW dependencies between and for the GIID property between all
    // the stores and the load
    if (!rawToLoadExists(loadOp, storeOps, nameAnalysis) &&
        storesAreGIIDOnLoad(loadOp, storeOps)) {
      removableLoads.insert(loadOp);
    }
  }
  if (removableLoads.empty())
    return;

  // Compute the set of stores that can be removed from the LSQ now that some
  // loads are out
}

std::unique_ptr<dynamatic::DynamaticPass>
dynamatic::createHandshakeMinimizeLSQUsage() {
  return std::make_unique<HandshakeMinimizeLSQUsagePass>();
}
