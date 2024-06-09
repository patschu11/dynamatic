// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: dynamatic-opt --handshake-optimize-bitwidths --remove-operation-names %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @addiBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: i8,
// CHECK-SAME:                           %[[VAL_1:.*]]: i32,
// CHECK-SAME:                           %[[VAL_2:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : i32 to i16
// CHECK:           %[[VAL_4:.*]] = arith.extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : i8 to i16
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : i16
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : i16
// CHECK:           end %[[VAL_6]] : i16
// CHECK:         }
handshake.func @addiBW(%arg0: i8, %arg1: i32, %start: none) -> i16 {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %res = arith.addi %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----

// CHECK-LABEL:   handshake.func @subiBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: i8,
// CHECK-SAME:                           %[[VAL_1:.*]]: i32,
// CHECK-SAME:                           %[[VAL_2:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : i32 to i16
// CHECK:           %[[VAL_4:.*]] = arith.extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : i8 to i16
// CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_4]], %[[VAL_3]] : i16
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : i16
// CHECK:           end %[[VAL_6]] : i16
// CHECK:         }
handshake.func @subiBW(%arg0: i8, %arg1: i32, %start: none) -> i16 {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %res = arith.subi %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----


// CHECK-LABEL:   handshake.func @muliBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: i8,
// CHECK-SAME:                           %[[VAL_1:.*]]: i32,
// CHECK-SAME:                           %[[VAL_2:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_0]] : i8 to i32
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : i16
// CHECK:           end %[[VAL_6]] : i16
// CHECK:         }
handshake.func @muliBW(%arg0: i8, %arg1: i32, %start: none) -> i16 {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %res = arith.muli %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----

// CHECK-LABEL:   handshake.func @andiBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: i8,
// CHECK-SAME:                           %[[VAL_1:.*]]: i32,
// CHECK-SAME:                           %[[VAL_2:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = arith.trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : i32 to i8
// CHECK:           %[[VAL_4:.*]] = arith.andi %[[VAL_0]], %[[VAL_3]] : i8
// CHECK:           %[[VAL_5:.*]] = arith.extui %[[VAL_4]] : i8 to i16
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : i16
// CHECK:           end %[[VAL_6]] : i16
// CHECK:         }
handshake.func @andiBW(%arg0: i8, %arg1: i32, %start: none) -> i16 {
  %ext0 = arith.extui %arg0 : i8 to i32
  %res = arith.andi %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----


// CHECK-LABEL:   handshake.func @oriBW(
// CHECK-SAME:                          %[[VAL_0:.*]]: i8,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32,
// CHECK-SAME:                          %[[VAL_2:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_0]] : i8 to i32
// CHECK:           %[[VAL_4:.*]] = arith.ori %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : i16
// CHECK:           end %[[VAL_6]] : i16
// CHECK:         }
handshake.func @oriBW(%arg0: i8, %arg1: i32, %start: none) -> i16 {
  %ext0 = arith.extui %arg0 : i8 to i32
  %res = arith.ori %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----


// CHECK-LABEL:   handshake.func @xoriBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: i8,
// CHECK-SAME:                           %[[VAL_1:.*]]: i32,
// CHECK-SAME:                           %[[VAL_2:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_3:.*]] = arith.extui %[[VAL_0]] : i8 to i32
// CHECK:           %[[VAL_4:.*]] = arith.xori %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK:           %[[VAL_6:.*]] = return %[[VAL_5]] : i16
// CHECK:           end %[[VAL_6]] : i16
// CHECK:         }
handshake.func @xoriBW(%arg0: i8, %arg1: i32, %start: none) -> i16 {
  %ext0 = arith.extui %arg0 : i8 to i32
  %res = arith.xori %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----

// CHECK-LABEL:   handshake.func @shliBW(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32,
// CHECK-SAME:                           %[[VAL_1:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = arith.trunci %[[VAL_0]] {handshake.bb = 0 : ui32} : i32 to i12
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] {handshake.bb = 0 : ui32} : i12 to i16
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_1]] {value = 4 : i32} : i32
// CHECK:           %[[VAL_5:.*]] = arith.trunci %[[VAL_4]] : i32 to i16
// CHECK:           %[[VAL_6:.*]] = arith.shli %[[VAL_3]], %[[VAL_5]] : i16
// CHECK:           %[[VAL_7:.*]] = return %[[VAL_6]] : i16
// CHECK:           end %[[VAL_7]] : i16
// CHECK:         }
handshake.func @shliBW(%arg0: i32, %start: none) -> i16 {
  %cst = handshake.constant %start {value = 4 : i32} : i32
  %res = arith.shli %arg0, %cst : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----

// CHECK-LABEL:   handshake.func @shrsiBW(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32,
// CHECK-SAME:                            %[[VAL_1:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = arith.trunci %[[VAL_0]] {handshake.bb = 0 : ui32} : i32 to i20
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]] {value = 4 : i32} : i32
// CHECK:           %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i32 to i20
// CHECK:           %[[VAL_5:.*]] = arith.shrsi %[[VAL_2]], %[[VAL_4]] : i20
// CHECK:           %[[VAL_6:.*]] = arith.trunci %[[VAL_5]] : i20 to i16
// CHECK:           %[[VAL_7:.*]] = return %[[VAL_6]] : i16
// CHECK:           end %[[VAL_7]] : i16
// CHECK:         }
handshake.func @shrsiBW(%arg0: i32, %start: none) -> i16 {
  %cst = handshake.constant %start {value = 4 : i32} : i32
  %res = arith.shrsi %arg0, %cst : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}

// -----

// CHECK-LABEL:   handshake.func @shruiBW(
// CHECK-SAME:                            %[[VAL_0:.*]]: i32,
// CHECK-SAME:                            %[[VAL_1:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_2:.*]] = arith.trunci %[[VAL_0]] {handshake.bb = 0 : ui32} : i32 to i20
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]] {value = 4 : i32} : i32
// CHECK:           %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i32 to i20
// CHECK:           %[[VAL_5:.*]] = arith.shrui %[[VAL_2]], %[[VAL_4]] : i20
// CHECK:           %[[VAL_6:.*]] = arith.trunci %[[VAL_5]] : i20 to i16
// CHECK:           %[[VAL_7:.*]] = return %[[VAL_6]] : i16
// CHECK:           end %[[VAL_7]] : i16
// CHECK:         }
handshake.func @shruiBW(%arg0: i32, %start: none) -> i16 {
  %cst = handshake.constant %start {value = 4 : i32} : i32
  %res = arith.shrui %arg0, %cst : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}


// -----

// CHECK-LABEL:   handshake.func @selectBW(
// CHECK-SAME:                             %[[VAL_0:.*]]: i8,
// CHECK-SAME:                             %[[VAL_1:.*]]: i32,
// CHECK-SAME:                             %[[VAL_2:.*]]: i1,
// CHECK-SAME:                             %[[VAL_3:.*]]: none, ...) -> i16 attributes {argNames = ["arg0", "arg1", "select", "start"], resNames = ["out0"]} {
// CHECK:           %[[VAL_4:.*]] = arith.trunci %[[VAL_1]] {handshake.bb = 0 : ui32} : i32 to i16
// CHECK:           %[[VAL_5:.*]] = arith.extsi %[[VAL_0]] {handshake.bb = 0 : ui32} : i8 to i16
// CHECK:           %[[VAL_6:.*]] = arith.select %[[VAL_2]], %[[VAL_5]], %[[VAL_4]] : i16
// CHECK:           %[[VAL_7:.*]] = return %[[VAL_6]] : i16
// CHECK:           end %[[VAL_7]] : i16
// CHECK:         }
handshake.func @selectBW(%arg0: i8, %arg1: i32, %select: i1, %start: none) -> i16 {
  %ext0 = arith.extsi %arg0 : i8 to i32
  %res = arith.select %select, %ext0, %arg1 : i32
  %trunc = arith.trunci %res : i32 to i16
  %returnVal = return %trunc : i16
  end %returnVal : i16
}
