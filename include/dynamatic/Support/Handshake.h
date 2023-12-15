//===- Handshake.h - Helpers for Handshake-level analysis -------*- C++ -*-===//
//
// Dynamatic is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares a couple "analysis functions" that may be helpful when
// working with Handshake-level IR. These may be useful in many different
// contexts and as such are not associated with any pass in particular.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "dynamatic/Support/LLVM.h"

namespace dynamatic {

/// Determines whether the given value has any "real" use i.e., a use which is
/// not the operand of a sink. If this function returns false for a given value
/// and one decides to erase the operation that defines it, one should keep in
/// mind that there may still be actual users of the value in the IR. In this
/// situation, using `eraseSinkUsers` in conjunction with this function will get
/// rid of all of the value's users.
bool hasRealUses(Value val);

/// Erases all sink operations that have the given value as operand.
void eraseSinkUsers(Value val);

/// Erases all sink operations that have the given value as operand. Uses the
/// rewriter to erase operations.
void eraseSinkUsers(Value val, PatternRewriter &rewriter);

/// Identifies the subset of the control operation's results that are part of
/// the control path to the LSQ interface. The control operations' results that
/// are not of type `NoneType` are ignored and will never be part of the
/// returned vector. Typically, one would call this function on a (lazy-)fork
/// directly providing a group allocation signal to the LSQ to inquire about
/// other fork results that would trigger other group allocations. The returned
/// values are guaranteed to be in the same order as the control operation's
/// results.
SmallVector<Value> getLSQControlPaths(circt::handshake::LSQOp lsqOp,
                                      Operation *ctrlOp);

/// Recursive function that determines whether a value is globally in-order
/// dependent on another value. In other words, the function answers the
/// following question: is `dependOn` always (i.e., independently of the path
/// through the DFG) involved in the determination of `val`?
///
/// The function achieves this by "bactracking" through the dataflow circuit
/// (from operations' results to operands) in search for the `dependOn` value.
/// The backtracking behavior is influenced by the type of the operation
/// traversed at each step. As it backtracks, the function keeps track of the
/// set of operations it has crossed on its path to be able to break cycles in
/// the DFG.
bool isGIID(Value dependOn, Value val,
            const mlir::DenseSet<Operation *> &path = {});

} // namespace dynamatic