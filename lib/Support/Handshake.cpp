//===- Handshake.cpp - Helpers for Handshake-level analysis -----*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements helpers for working with Handshake-level IR.
//
//===----------------------------------------------------------------------===//

#include "dynamatic/Support/Handshake.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/Attribute.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

void MemoryOpLowering::recordReplacement(Operation *oldOp, Operation *newOp,
                                         bool forwardInterface) {
  copyAttr<MemDependenceArrayAttr>(oldOp, newOp);
  if (forwardInterface)
    copyAttr<MemInterfaceAttr>(oldOp, newOp);
  nameChanges[namer.getName(oldOp)] = namer.getName(newOp);
}

bool MemoryOpLowering::renameDependencies(Operation *topLevelOp) {
  MLIRContext *ctx = topLevelOp->getContext();
  bool anyChange = false;
  topLevelOp->walk([&](Operation *memOp) {
    // We only care about supported load/store memory accesses
    if (!isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp,
             affine::AffineStoreOp>(memOp))
      return;

    // Read potential memory dependencies stored on the memory operation
    auto oldMemDeps = getUniqueAttr<MemDependenceArrayAttr>(memOp);
    if (!oldMemDeps)
      return;

    // Copy memory dependence attributes one-by-one, replacing the name of
    // replaced destination memory operations along the way if necessary
    SmallVector<MemDependenceAttr> newMemDeps;
    for (MemDependenceAttr oldDep : oldMemDeps.getDependencies()) {
      StringRef oldName = oldDep.getDstAccess();
      auto replacedName = nameChanges.find(oldName);
      bool opWasReplaced = replacedName != nameChanges.end();
      anyChange |= opWasReplaced;
      if (opWasReplaced) {
        StringAttr newName = StringAttr::get(ctx, replacedName->second);
        newMemDeps.push_back(MemDependenceAttr::get(
            ctx, newName, oldDep.getLoopDepth(), oldDep.getComponents()));
      } else {
        newMemDeps.push_back(oldDep);
      }
    }
    setUniqueAttr(memOp, MemDependenceArrayAttr::get(ctx, newMemDeps));
  });

  return anyChange;
}

bool dynamatic::hasRealUses(Value val) {
  return llvm::any_of(val.getUsers(), [&](Operation *user) {
    return !isa<handshake::SinkOp>(user);
  });
}

void dynamatic::eraseSinkUsers(Value val) {
  for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
    if (isa<handshake::SinkOp>(user))
      user->erase();
  }
}

void dynamatic::eraseSinkUsers(Value val, PatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(val.getUsers())) {
    if (isa<handshake::SinkOp>(user))
      rewriter.eraseOp(user);
  }
}

SmallVector<Value> dynamatic::getLSQControlPaths(handshake::LSQOp lsqOp,
                                                 Operation *ctrlOp) {
  // Accumulate all outputs of the control operation that are part of the memory
  // control network
  SmallVector<Value> controlValues;
  // List of control channels to explore, starting from the control operation's
  // results
  SmallVector<Value, 4> controlChannels;
  // Set of control operations already explored from the control operation's
  // results (to avoid looping in the dataflow graph)
  SmallPtrSet<Operation *, 4> controlOps;
  for (OpResult res : ctrlOp->getResults()) {
    // We only care for control-only channels
    if (!isa<NoneType>(res.getType()))
      continue;

    // Reset the list of control channels to explore and the list of control
    // operations that we have already visited
    controlChannels.clear();
    controlOps.clear();

    controlChannels.push_back(res);
    controlOps.insert(ctrlOp);
    do {
      Value val = controlChannels.pop_back_val();
      for (Operation *succOp : val.getUsers()) {
        // Make sure that we do not loop forever over the same control
        // operations
        if (auto [_, newOp] = controlOps.insert(succOp); !newOp)
          continue;

        if (succOp == lsqOp) {
          // We have found a control path triggering a different group
          // allocation to the LSQ, add it to our list
          controlValues.push_back(res);
          break;
        }
        llvm::TypeSwitch<Operation *, void>(succOp)
            .Case<handshake::ConditionalBranchOp, handshake::BranchOp,
                  handshake::MergeOp, handshake::MuxOp, handshake::ForkOp,
                  handshake::LazyForkOp, handshake::BufferOp>([&](auto) {
              // If the successor just propagates the control path, add
              // all its results to the list of control channels to
              // explore
              for (OpResult succRes : succOp->getResults())
                controlChannels.push_back(succRes);
            })
            .Case<handshake::ControlMergeOp>(
                [&](handshake::ControlMergeOp cmergeOp) {
                  // Only the control merge's data output forwards the input
                  controlChannels.push_back(cmergeOp.getResult());
                });
      }
    } while (!controlChannels.empty());
  }

  return controlValues;
}

bool dynamatic::isGIID(Value dependOn, Value val,
                       const DenseSet<Operation *> &path) {
  if (dependOn == val)
    return true;

  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  if (path.contains(defOp)) {
    // If we are encountering an operation for the second time it means that we
    // went through an entire CFG cycle, which implies that there was a
    // merge-like operation on our path that is computing the conjunction of
    // this function's results on all its data inputs. In that case, the merge
    // inputs coming from outside the cycle determine whether the entire cycle
    // depends on the value, so we return true to not falsify the conjunction.
    // If merge input coming from outside the cycle do not depend on the value,
    // the function's top-level call will still return false
    return true;
  }

  // Recursively call the function with a new value as second argument (meant to
  // be an operand to the defining operation) and adding the defining operation
  // to the path.
  auto recGIID = [&](Value newVal) -> bool {
    DenseSet<Operation *> newPath(path);
    newPath.insert(defOp);
    return isGIID(dependOn, newVal, newPath);
  };

  // The backtracking logic depends on the type of the defining operation
  return llvm::TypeSwitch<Operation *, bool>(defOp)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condBrOp) {
            // The data operand or the condition operand must depend on the
            // value
            return recGIID(condBrOp.getDataOperand()) ||
                   recGIID(condBrOp.getConditionOperand());
          })
      .Case<handshake::MergeOp, handshake::ControlMergeOp>([&](auto) {
        // All data inputs must depend on the value
        return llvm::all_of(defOp->getOperands(), [&](Value mergeLikeOprd) {
          return recGIID(mergeLikeOprd);
        });
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // If the select operand depends on the value, then the mux depends on
        // the value
        if (recGIID(muxOp.getSelectOperand()))
          return true;
        // Otherwise, all data inputs must depend on the value
        return llvm::all_of(defOp->getOperands(), [&](Value mergeLikeOprd) {
          return recGIID(mergeLikeOprd);
        });
      })
      .Case<handshake::DynamaticReturnOp>([&](auto) {
        // Just recurse the call on the return operand corresponding to the
        // value
        return recGIID(
            defOp->getOperand(cast<OpResult>(val).getResultNumber()));
      })
      .Case<handshake::MCLoadOp, handshake::LSQLoadOp>([&](auto) {
        auto loadOp = cast<handshake::LoadOpInterface>(defOp);
        if (loadOp.getDataOutput() != val)
          return false;

        // If the address operand depends on the value then the data result
        // depends on the value
        return recGIID(loadOp.getAddressInput());
      })
      .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
        // Similarly to the mux, if the select operand depends on the value,
        // then the select depends on the value
        if (recGIID(selectOp.getCondition()))
          return true;

        // The select's true value and false value must depend on the value
        return recGIID(selectOp.getTrueValue()) &&
               recGIID(selectOp.getFalseValue());
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::BufferOp,
            handshake::BranchOp, arith::AddIOp, arith::AndIOp, arith::CmpIOp,
            arith::DivSIOp, arith::DivUIOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::MulIOp, arith::OrIOp, arith::RemUIOp, arith::RemSIOp,
            arith::ShLIOp, arith::ShRUIOp, arith::SIToFPOp, arith::SubIOp,
            arith::TruncIOp, arith::UIToFPOp, arith::XOrIOp, arith::AddFOp,
            arith::CmpFOp, arith::DivFOp, arith::ExtFOp, arith::MulFOp,
            arith::RemFOp, arith::SubFOp, arith::TruncFOp>([&](auto) {
        // At least one operand must depend on the value
        return llvm::any_of(defOp->getOperands(),
                            [&](Value oprd) { return recGIID(oprd); });
      })
      .Default([&](auto) { return false; });
}
