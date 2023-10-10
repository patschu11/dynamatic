set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(export-dot-cfdfc
  export-dot-cfdfc.cpp
)

llvm_update_compile_flags(export-dot-cfdfc)
target_link_libraries(export-dot-cfdfc
  PRIVATE
  DynamaticSupport
  DynamaticTransforms

  CIRCTHandshake
  CIRCTHandshakeTransforms

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRMemRefDialect
  )
