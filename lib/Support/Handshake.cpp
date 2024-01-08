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
#include "circt/Support/BackedgeBuilder.h"
#include "dynamatic/Support/Attribute.h"
#include "dynamatic/Support/CFG.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace dynamatic;

//===----------------------------------------------------------------------===//
// MemoryOpLowering
//===----------------------------------------------------------------------===//

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
             affine::AffineStoreOp, handshake::LoadOpInterface,
             handshake::StoreOpInterface>(memOp))
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

//===----------------------------------------------------------------------===//
// MemoryInterfaceBuilder
//===----------------------------------------------------------------------===//

void MemoryInterfaceBuilder::addMCPort(Operation *memOp) {
  std::optional<unsigned> bb = getLogicBB(memOp);
  assert(bb && "MC port must belong to basic block");
  if (isa<handshake::MCLoadOp>(memOp)) {
    ++mcNumLoads;
  } else {
    assert(isa<handshake::MCStoreOp>(memOp) && "invalid MC port");
  }
  mcPorts[*bb].push_back(memOp);
}

void MemoryInterfaceBuilder::addLSQPort(unsigned group, Operation *memOp) {
  if (isa<handshake::LSQLoadOp>(memOp)) {
    ++lsqNumLoads;
  } else {
    assert(isa<handshake::LSQStoreOp>(memOp) && "invalid LSQ port");
  }
  lsqPorts[group].push_back(memOp);
}

LogicalResult MemoryInterfaceBuilder::instantiateInterfaces(
    mlir::PatternRewriter &rewriter, circt::handshake::MemoryControllerOp &mcOp,
    circt::handshake::LSQOp &lsqOp) {

  // Determine interfaces' inputs
  InterfaceInputs inputs;
  if (failed(determineInterfaceInputs(inputs, rewriter)))
    return failure();
  if (inputs.mcInputs.empty() && inputs.lsqInputs.empty())
    return success();

  mcOp = nullptr;
  lsqOp = nullptr;

  rewriter.setInsertionPointToStart(&funcOp.front());
  Location loc = memref.getLoc();

  if (!inputs.mcInputs.empty() && inputs.lsqInputs.empty()) {
    // We only need a memory controller
    mcOp = rewriter.create<handshake::MemoryControllerOp>(
        loc, memref, inputs.mcInputs, inputs.mcBlocks, mcNumLoads);
  } else if (inputs.mcInputs.empty() && !inputs.lsqInputs.empty()) {
    // We only need an LSQ
    lsqOp = rewriter.create<handshake::LSQOp>(
        loc, memref, inputs.lsqInputs, inputs.lsqGroupSizes, lsqNumLoads);
  } else {
    // We need a MC and an LSQ. They need to be connected with 4 new channels
    // so that the LSQ can forward its loads and stores to the MC. We need
    // load address, store address, and store data channels from the LSQ to
    // the MC and a load data channel from the MC to the LSQ
    MemRefType memrefType = memref.getType().cast<MemRefType>();

    // Create 3 backedges (load address, store address, store data) for the MC
    // inputs that will eventually come from the LSQ.
    BackedgeBuilder edgeBuilder(rewriter, loc);
    Backedge ldAddr = edgeBuilder.get(rewriter.getIndexType());
    Backedge stAddr = edgeBuilder.get(rewriter.getIndexType());
    Backedge stData = edgeBuilder.get(memrefType.getElementType());
    inputs.mcInputs.push_back(ldAddr);
    inputs.mcInputs.push_back(stAddr);
    inputs.mcInputs.push_back(stData);

    // Create the memory controller, adding 1 to its load count so that it
    // generates a load data result for the LSQ
    mcOp = rewriter.create<handshake::MemoryControllerOp>(
        loc, memref, inputs.mcInputs, inputs.mcBlocks, mcNumLoads + 1);

    // Add the MC's load data result to the LSQ's inputs and create the LSQ,
    // passing a flag to the builder so that it generates the necessary
    // outputs that will go to the MC
    inputs.lsqInputs.push_back(mcOp.getMemOutputs().back());
    lsqOp = rewriter.create<handshake::LSQOp>(
        loc, mcOp, inputs.lsqInputs, inputs.lsqGroupSizes, lsqNumLoads);

    // Resolve the backedges to fully connect the MC and LSQ
    ValueRange lsqMemResults = lsqOp.getMemOutputs().take_back(3);
    ldAddr.setValue(lsqMemResults[0]);
    stAddr.setValue(lsqMemResults[1]);
    stData.setValue(lsqMemResults[2]);
  }

  // At this point, all load ports are missing their second operand which is the
  // data value coming from a memory interface back to the port
  if (mcOp)
    addMemDataResultToLoads(mcPorts, mcOp);
  if (lsqOp)
    addMemDataResultToLoads(lsqPorts, lsqOp);

  return success();
}

SmallVector<Value, 2>
MemoryInterfaceBuilder::getMemResultsToInterface(Operation *memOp) {
  // For loads, address output go to memory
  if (auto loadOp = dyn_cast<handshake::LoadOpInterface>(memOp))
    return SmallVector<Value, 2>{loadOp.getAddressOutput()};

  // For stores, all outputs (address and data) go to memory
  auto storeOp = dyn_cast<handshake::StoreOpInterface>(memOp);
  assert(storeOp && "input operation must either be load or store");
  return SmallVector<Value, 2>{storeOp->getResults()};
}

Value MemoryInterfaceBuilder::getMCControl(Value ctrl, unsigned numStores,
                                           PatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(ctrl.getDefiningOp());
  handshake::ConstantOp cstOp = rewriter.create<handshake::ConstantOp>(
      ctrl.getLoc(), rewriter.getI32Type(),
      rewriter.getI32IntegerAttr(numStores), ctrl);
  inheritBBFromValue(ctrl, cstOp);
  return cstOp.getResult();
}

void MemoryInterfaceBuilder::setLoadDataOperand(
    handshake::LoadOpInterface loadOp, Value dataIn) {
  SmallVector<Value, 2> operands;
  operands.push_back(loadOp->getOperand(0));
  operands.push_back(dataIn);
  loadOp->setOperands(operands);
}

LogicalResult MemoryInterfaceBuilder::determineInterfaceInputs(
    InterfaceInputs &inputs, mlir::PatternRewriter &rewriter) {

  // Determine LSQ inputs
  for (auto [group, lsqGroupOps] : lsqPorts) {
    // First, determine the group's control signal, which is dictated by the BB
    // of the first memory port in the group
    Operation *firstOpInGroup = lsqGroupOps.front();
    std::optional<unsigned> block = getLogicBB(firstOpInGroup);
    if (!block)
      return firstOpInGroup->emitError() << "LSQ port must belong to a BB.";
    Value groupCtrl = getCtrl(*block);
    if (!groupCtrl)
      return failure();
    inputs.lsqInputs.push_back(groupCtrl);

    // Them, add all memory port results that go the interface to the list of
    // LSQ inputs
    for (Operation *lsqOp : lsqGroupOps)
      llvm::copy(getMemResultsToInterface(lsqOp),
                 std::back_inserter(inputs.lsqInputs));

    // Add the size of the group to our list
    inputs.lsqGroupSizes.push_back(lsqGroupOps.size());
  }

  if (mcPorts.empty())
    return success();

  // The MC needs control signals from all blocks containing store ports
  // connected to an LSQ, since these requests end up being forwarded to the MC,
  // so we need to know the number of LSQ stores per basic block
  DenseMap<unsigned, unsigned> lsqStoresPerBlock;
  for (auto [_, lsqGroupOps] : lsqPorts) {
    for (Operation *lsqOp : lsqGroupOps) {
      if (isa<handshake::LSQStoreOp>(lsqOp)) {
        std::optional<unsigned> block = getLogicBB(lsqOp);
        if (!block)
          return lsqOp->emitError() << "LSQ port must belong to a BB.";
        ++lsqStoresPerBlock[*block] += 1;
      }
    }
  }

  // Inputs from blocks that have at least one direct load/store access port to
  // the MC are added to the future MC's operands first
  for (auto &[block, mcBlockOps] : mcPorts) {
    // Count the total number of stores in the block, either directly connected
    // to the MC or going through an LSQ
    unsigned numStoresInBlock = lsqStoresPerBlock.lookup(block);
    for (Operation *memOp : mcBlockOps) {
      if (isa<handshake::MCStoreOp>(memOp))
        ++numStoresInBlock;
    }

    // Blocks with at least one store need to provide a control signal fed
    // through a constant indicating the number of stores in the block
    if (numStoresInBlock > 0) {
      Value blockCtrl = getCtrl(block);
      if (!blockCtrl)
        return failure();
      inputs.mcInputs.push_back(
          getMCControl(blockCtrl, numStoresInBlock, rewriter));
    }

    // Traverse the list of memory operations in the block once more and
    // accumulate memory inputs coming from the block
    for (Operation *mcOp : mcBlockOps)
      llvm::copy(getMemResultsToInterface(mcOp),
                 std::back_inserter(inputs.mcInputs));

    inputs.mcBlocks.push_back(block);
  }

  // Control ports from blocks which do not have memory ports directly
  // connected to the MC but from which the LSQ will forward store requests from
  // are then added to the future MC's operands
  for (auto &[lsqBlock, numStores] : lsqStoresPerBlock) {
    // We only need to do something if the block has stores that have not yet
    // been accounted for
    if (mcPorts.contains(lsqBlock) || numStores == 0)
      continue;

    // Identically to before, blocks with stores need a cntrol signal
    Value blockCtrl = getCtrl(lsqBlock);
    if (!blockCtrl)
      return failure();
    inputs.mcInputs.push_back(getMCControl(blockCtrl, numStores, rewriter));

    inputs.mcBlocks.push_back(lsqBlock);
  }

  return success();
}

Value MemoryInterfaceBuilder::getCtrl(unsigned block) {
  auto groupCtrl = ctrlVals.find(block);
  if (groupCtrl == ctrlVals.end()) {
    llvm::errs() << "Cannot determine control signal for BB " << block << "\n";
    return nullptr;
  }
  return groupCtrl->second;
}

void MemoryInterfaceBuilder::addMemDataResultToLoads(InterfacePorts &ports,
                                                     Operation *memIfaceOp) {
  unsigned resIdx = 0;
  for (auto &[_, memGroupOps] : ports) {
    for (Operation *memOp : memGroupOps) {
      if (auto loadOp = dyn_cast<handshake::LoadOpInterface>(memOp))
        setLoadDataOperand(loadOp, memIfaceOp->getResult(resIdx++));
    }
  }
}

//===----------------------------------------------------------------------===//
// Misc functions
//===----------------------------------------------------------------------===//

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

namespace {
enum class GIIDStatus { FAIL_OFF_PATH, FAIL_ON_PATH, SUCCEED };
} // namespace

static GIIDStatus
foldGIIDStatusOr(const std::function<GIIDStatus(Value)> &callback,
                 ValueRange values) {
  if (values.empty())
    return GIIDStatus::FAIL_OFF_PATH;

  // To succeed, it is enough that one operand reaches the predecessor on the
  // CFG path. If all operands are off path, then report an "off-path
  // failure". If no operand reaches the predecessor but at least one is on
  // path, report an "on-path failure"
  GIIDStatus stat = GIIDStatus::FAIL_OFF_PATH;
  for (Value newVal : values) {
    switch (callback(newVal)) {
    case GIIDStatus::FAIL_OFF_PATH:
      // Do nothing here, so that if all operands are off path we end up
      // reporting an "off-path failure"
      break;
    case GIIDStatus::FAIL_ON_PATH:
      // Unless another operand reaches the predecessor, we will end up
      // reporting an "on-path failure"
      stat = GIIDStatus::FAIL_ON_PATH;
      break;
    case GIIDStatus::SUCCEED:
      // Early return when one of the datapaths reaches the predecessor
      return GIIDStatus::SUCCEED;
    }
  }
  return stat;
}

static GIIDStatus
foldGIIDStatusAnd(const std::function<GIIDStatus(Value)> &callback,
                  ValueRange values) {
  if (values.empty())
    return GIIDStatus::FAIL_OFF_PATH;

  // To succeed, at least one operand must reach the predecessor on the CFG
  // path, and none must fail to reach the predecessor on the path. If all
  // operands are off path, then report an "off-path failure"
  GIIDStatus stat = GIIDStatus::FAIL_OFF_PATH;
  for (Value newVal : values) {
    switch (callback(newVal)) {
    case GIIDStatus::FAIL_OFF_PATH:
      // Do nothing here, so that if all operands are off path we end up
      // reporting an "off-path failure"
      break;
    case GIIDStatus::FAIL_ON_PATH:
      // All datapaths on the CFG path must connect to the predecessor. Early
      // return when we fail on the path.
      return GIIDStatus::FAIL_ON_PATH;
    case GIIDStatus::SUCCEED:
      // Unless another operand is on the path but fails to reach the
      // predecessor, this call will succeed
      stat = GIIDStatus::SUCCEED;
      break;
    }
  }
  return stat;
}

static GIIDStatus isGIIDRec(Value predecessor, OpOperand &oprd, CFGPath &path,
                            unsigned pathIdx) {
  Value val = oprd.get();
  if (predecessor == val)
    return GIIDStatus::SUCCEED;

  // The defining operation must exist, otherwise it means we have reached
  // function arguments without encountering the predecessor value. It must also
  // belong to a block
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return GIIDStatus::FAIL_ON_PATH;
  std::optional<unsigned> defBB = getLogicBB(defOp);
  if (!defBB)
    return GIIDStatus::FAIL_ON_PATH;

  if (isBackedge(val, oprd.getOwner())) {
    // Backedges always indicate transitions from one block on the path to its
    // predecessor

    // If we move past the first block in the path, so it's an "on-path failure"
    if (pathIdx == 0)
      return GIIDStatus::FAIL_ON_PATH;

    // The previous block in the path must be the one the defining operation
    // belongs to, otherwise it's an "off-path failure"
    if (path[pathIdx - 1] != defBB)
      return GIIDStatus::FAIL_OFF_PATH;
    --pathIdx;
  } else {
    // The defining operation must be somewhere earlier in the path than before.
    // We allow the path to "jump over" BBs, since datapaths of optimized
    // circuits will sometimes skip BBs entirely

    bool foundOnPath = false;
    for (size_t newIdx = pathIdx + 1; newIdx > 0; --newIdx) {
      if (path[newIdx - 1] == *defBB) {
        foundOnPath = true;
        pathIdx = newIdx - 1;
        break;
      }
    }
    if (!foundOnPath) {
      // If we had previously reached the block where the predecessor is defined
      // and moved past it, the failure is "on path". If we failed earlier, the
      // failure is "off" path.
      /// NOTE: Is that last part true? Is it possible to jump over the block
      /// defining the predecessor and still be on path?
      return pathIdx == 0 ? GIIDStatus::FAIL_ON_PATH
                          : GIIDStatus::FAIL_OFF_PATH;
    }
  }

  // Recursively calls the function with a new value as second argument (meant
  // to be an operand of the defining operation identified above) and changing
  // the current defining operation to be the current one
  auto recurse = [&](Value newVal) -> GIIDStatus {
    for (OpOperand &oprd : defOp->getOpOperands()) {
      if (oprd.get() == newVal)
        return isGIIDRec(predecessor, oprd, path, pathIdx);
    }
    llvm_unreachable("recursive call should be on operand of defining op");
  };

  // The backtracking logic depends on the type of the defining operation
  return llvm::TypeSwitch<Operation *, GIIDStatus>(defOp)
      .Case<handshake::ConditionalBranchOp>(
          [&](handshake::ConditionalBranchOp condBrOp) {
            // The data operand or the condition operand must depend on the
            // predecessor
            return foldGIIDStatusAnd(recurse, condBrOp->getOperands());
          })
      .Case<handshake::MergeOp, handshake::ControlMergeOp>([&](auto) {
        // The data input on the path must depend on the predecessor
        return foldGIIDStatusAnd(recurse, defOp->getOperands());
      })
      .Case<handshake::MuxOp>([&](handshake::MuxOp muxOp) {
        // If the select operand depends on the predecessor, then the mux
        // depends on the predecessor
        if (recurse(muxOp.getSelectOperand()) == GIIDStatus::SUCCEED)
          return GIIDStatus::SUCCEED;

        // Otherwise, data inputs on the path must depend on the predecessor
        return foldGIIDStatusAnd(recurse, defOp->getOperands());
      })
      .Case<handshake::DynamaticReturnOp>([&](auto) {
        // Just recurse the call on the return operand corresponding to the
        // value
        Value oprd = defOp->getOperand(cast<OpResult>(val).getResultNumber());
        return recurse(oprd);
      })
      .Case<handshake::MCLoadOp, handshake::LSQLoadOp>([&](auto) {
        auto loadOp = cast<handshake::LoadOpInterface>(defOp);
        if (loadOp.getDataOutput() != val)
          return GIIDStatus::FAIL_ON_PATH;

        // If the address operand depends on the predecessor then the data
        // result depends on the predecessor
        return recurse(loadOp.getAddressInput());
      })
      .Case<arith::SelectOp>([&](arith::SelectOp selectOp) {
        // Similarly to the mux, if the select operand depends on the
        // predecessor, then the select depends on the predecessor
        if (recurse(selectOp.getCondition()) == GIIDStatus::SUCCEED)
          return GIIDStatus::SUCCEED;

        // The select's true value or false value must depend on the predecessor
        ValueRange values{selectOp.getTrueValue(), selectOp.getFalseValue()};
        return foldGIIDStatusAnd(recurse, values);
      })
      .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::BufferOp,
            handshake::BranchOp, arith::AddIOp, arith::AndIOp, arith::CmpIOp,
            arith::DivSIOp, arith::DivUIOp, arith::ExtSIOp, arith::ExtUIOp,
            arith::MulIOp, arith::OrIOp, arith::RemUIOp, arith::RemSIOp,
            arith::ShLIOp, arith::ShRUIOp, arith::SIToFPOp, arith::SubIOp,
            arith::TruncIOp, arith::UIToFPOp, arith::XOrIOp, arith::AddFOp,
            arith::CmpFOp, arith::DivFOp, arith::ExtFOp, arith::MulFOp,
            arith::RemFOp, arith::SubFOp, arith::TruncFOp>([&](auto) {
        // At least one operand must depend on the predecessor
        return foldGIIDStatusOr(recurse, defOp->getOperands());
      })
      .Default([&](auto) {
        // To err on the conservative side, produce the most terminating kind of
        // failure on encoutering an unsupported operation
        return GIIDStatus::FAIL_ON_PATH;
      });
}

bool dynamatic::isGIID(Value predecessor, OpOperand &oprd, CFGPath &path) {
  assert(path.size() >= 2 && "path must have at least two blocks");
  return isGIIDRec(predecessor, oprd, path, path.size() - 1) ==
         GIIDStatus::SUCCEED;
}
