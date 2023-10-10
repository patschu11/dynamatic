# Run all integration tests available on the repository

# Indicate the path to your legacy Dynamatic install here (required for write-hdl)
set-legacy-path   ../dynamatic-utils/legacy-dynamatic/dhls/etc/dynamatic

# threshold
set-src           integration-test/src/polyn_mult/polyn_mult.c
synthesize        
write-hdl
simulate

# triangular
set-src           integration-test/src/stencil_2d/stencil_2d.c
synthesize        
write-hdl
simulate

# Exit the frontend
exit 