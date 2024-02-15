# Using Dynamatic

> [!IMPORTANT]
> This exercise, meant to be followed as part of the *Dynamatic Reloaded* workshop @ FPGA'24, is a condensed version of the [*Introduction to Dynamatic* tutorial's first chapter](../Tutorials/Introduction/UsingDynamatic.md) that is part of Dynamatic's documentation. To get more insights into everything happening at each step of the exercise, you are invited to check out the full tutorial. All shell commands throughout the tutorial must be executed from Dynamatic's top-level directory.

This exercise walks you through the compilation of a simple kernel function written in C into an equivalent VHDL design, the functional verification of the resulting dataflow circuit using Modelsim, and the latter's visualization using our custom interactive dataflow visualizer. 

## The source code

In this tutorial, we will tranform the following C function (the *kernel*, in DHLS jargon) into a dataflow circuit (source available [here](../../tutorials/Introduction/UsingDynamatic/loop_accumulate.c)).

```c
// The number of loop iterations
#define N 10

// The kernel under consideration
unsigned loop_accumulate(in_int_t a[N]) {
  unsigned x = 2;
  for (unsigned i = 0; i < N; ++i) {
    if (a[i] == 0)
      x = x * x;
  }
  return x;
}
```

This simple kernel multiplies a number by itself at each iteration of a simple loop from 0 to any number `N` where the corresponding element of an array equals 0. The function returns the accumulated value after the loop exits.

## Using Dynamatic's frontend

### Interactive mode

We will now use Dynamatic's frontend in interactive mode to compile the `loop_accumulate` kernel into a VHDL design in a couple of simple commands. In a terminal, from Dynamatic's top-level folder, run the following.

```sh
./bin/dynamatic
```

This will print the frontend's header and display a prompt where you can start inputting your first command.

```
================================================================================
============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
==================== EPFL-LAP - <release> | <release-date> =====================
================================================================================

dynamatic> # Input your command here
```

First, we must provide the frontend with the path to the C source file under consideration. Ours is located at [`tutorials/Introduction/UsingDynamatic/loop_accumulate.c`](../../tutorials/Introduction/UsingDynamatic/loop_accumulate.c), so input the following command into the frontend.

```
dynamatic> set-src tutorials/Introduction/UsingDynamatic/loop_accumulate.c
```

The first processing step is compilation, which will transform the C kernel into a low-level IR representation that we can later export to VHDL. To compile the C function, simply input `compile`.

```
dynamatic> set-src tutorials/Introduction/UsingDynamatic/loop_accumulate.c
dynamatic> compile
[INFO] Compiled source to affine
[INFO] Ran memory analysis
[INFO] Compiled affine to scf
[INFO] Compiled scf to cf
[INFO] Applied standard transformations to cf
[INFO] Applied Dynamatic transformations to cf
[INFO] Compiled cf to handshake
[INFO] Applied transformations to handshake
[INFO] Built kernel for profiling
[INFO] Ran kernel for profiling
[INFO] Profiled cf-level
[INFO] Running smart buffer placement
[INFO] Placed smart buffers
[INFO] Canonicalized handshake
[INFO] Created visual DOT
[INFO] Converted visual DOT to PNG
[INFO] Created loop_accumulate DOT
[INFO] Converted loop_accumulate DOT to PNG
[INFO] Compilation succeeded
```

> [!TIP]
> The `compile` command creates a static visual representation of the generated dataflow circuit in `tutorials/Introduction/UsingDynamatic/out/comp/visual.png`. Open it and study the graph corresponding to the dataflow circuit. Nodes in the graph represent dataflow components connected by dataflow channels (edges).

Now generate the VHDL design using the `write-hdl` command.

```
...
[INFO] Compilation succeeded

dynamatic> write-hdl
[INFO] Converted DOT to VHDL
[INFO] HDL generation succeeded
```

The VHDL design can finally be co-simulated along the C function on Modelsim to verify that their behavior matches using the `simulate` command.

```
...
[INFO] HDL generation succeeded

dynamatic> simulate
[INFO] Built kernel for IO gen.
[INFO] Ran kernel for IO gen.
[INFO] Launching Modelsim simulation
[INFO] Simulation succeeded
```

That's it, you have successfully synthesized your first dataflow circuit from C code and functionnaly verified it using Dynamatic! At this point, you can quit the Dynamatic frontend by inputting the `exit` command.

### Non-interactive mode

Note that the frontend can be used non-interactively by writing the list of commands one wishes to execute in a text file. You can replay the entire previous subsection at once by running the following from Dynamatic's top-level folder.

```sh
./bin/dynamatic --run tutorials/Introduction/UsingDynamatic/frontend-script.sh
```

In the last section of this tutorial, we will take a closer look at the actual circuit that was generated by Dynamatic and visualize its execution interactively. 

## Visualizing the resulting dataflow circuit

To generate the information needed by the visualizer, re-open the frontend, re-set the source, then input the `visualize` command.

```
$ ./bin/dynamatic
================================================================================
============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
==================== EPFL-LAP - <release> | <release-date> =====================
================================================================================

dynamatic> set-src tutorials/Introduction/UsingDynamatic/loop_accumulate.c
dynamatic> visualize
[INFO] Generated channel changes
[INFO] Added positioning info. to DOT

dynamatic> exit

Goodbye!
```

Now launch the dataflow visualizer from Dynamatic's top-level folder.
```sh
./bin/dataflow-visualizer
```

On the main menu, the visualizer prompts you to indicate the path to a DOT file and a CSV file to render the visualization. They are located respectively at `tutorials/Introduction/UsingDynamatic/out/visual/loop_accumulate.dot` and `tutorials/Introduction/UsingDynamatic/out/visual/sim.csv`. Once you have selected them, click `Draw graph`. You should now see a visual representation of the dataflow circuit you just synthesized. In this example, the original source code had 5 basic blocks, which are transcribed here in 5 labeled rectangular boxes.

> [!TIP]
> Two of these basic blocks represent the start and end of the kernel before and after the loop, respectively. The other 3 hold the loop's logic. Try to identify which is which from the nature of the nodes and from their connections. Consider that the loop may have been slightly transformed by Dynamatic to optimize the resulting circuit.

> [!TIP]
> Observe the circuit executes using the interactive controls at the bottom of the window. On cycle 6, for example, you can see that tokens are transferred on both input channels of `muli0` in `block2`. Try to infer the multiplier's latency by looking at its output channel in the next execution cycles. Then, try to track that output token through the circuit to see where it can end up. Study the execution till you get an understanding of how tokens flow inside the loop and of how the conditional multiplication influences the latency of each loop iteration.

## Conclusion

Congratulations on reaching the end of this tutorial! You now know how to use Dynamatic to compile C kernels into functional dataflow circuits, then visualize these circuits to better understand them and identify potential optimization opportunities. In the [next exercise](ModifyingDynamatic.md), we will identify one such opportunity and write a small transformation pass in C++ to implement our desired optimization, before finally verifying its behavior using the dataflow visualizer.
