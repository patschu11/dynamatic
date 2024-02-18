# Using Dynamatic

> [!NOTE]
> This exercise, meant to be followed as part of the *Dynamatic Reloaded* workshop @ FPGA'24, is a condensed version of the [*Introduction to Dynamatic* tutorial's first chapter](../Tutorials/Introduction/UsingDynamatic.md) that is part of Dynamatic's documentation. To get more insights into everything happening at each step of the exercise, you are invited to check out the full tutorial. All shell commands throughout the exercise must be executed from Dynamatic's top-level directory.

Throughout the exercise, you will see the following two types of blocks.
> [!IMPORTANT]
> *Important* blocks indicate when you need to do something before continuing the exercise.

> [!TIP]
> *Tip* blocks indicate something that you should spend some time exploring on your own to better understand the exercise's content.

This first exercise walks you through the compilation of a simple kernel function written in C into an equivalent VHDL design, the functional verification of the resulting dataflow circuit using Modelsim, and the latter's visualization using our custom interactive dataflow visualizer. 

## The source code

In this tutorial, we will tranform the following C function (the *kernel*, in DHLS jargon) into a dataflow circuit (source available [here](../../tutorials/Introduction/Ch1/loop_accumulate.c)).

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

We will now use Dynamatic's frontend in interactive mode to compile the `loop_accumulate` kernel into a VHDL design in a couple of simple commands. 

> [!IMPORTANT]
> In a terminal, from Dynamatic's top-level folder, run the following.
> ```sh
> ./bin/dynamatic
> ```

This will print the frontend's header and display a prompt where you can start inputting your first command.

```
================================================================================
============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
==================== EPFL-LAP - <release> | <release-date> =====================
================================================================================

dynamatic> # Input your command here
```

First, we must provide the frontend with the path to the C source file under consideration. Ours is located at [`tutorials/Introduction/Ch1/loop_accumulate.c`](../../tutorials/Introduction/Ch1/loop_accumulate.c).


> [!IMPORTANT]
> Set the source by inputting the following `set-src <path>` command into the frontend.
> ```
> dynamatic> set-src tutorials/Introduction/Ch1/loop_accumulate.c
> ```

The first processing step is compilation, which will transform the C kernel into a low-level IR representation that we can later export to VHDL.

> [!IMPORTANT]
> To compile the C function, simply write `compile`.
> ```
> dynamatic> compile
> [INFO] Compiled source to affine
> [INFO] Ran memory analysis
> [INFO] Compiled affine to scf
> [INFO] Compiled scf to cf
> [INFO] Applied standard transformations to cf
> [INFO] Applied Dynamatic transformations to cf
> [INFO] Compiled cf to handshake
> [INFO] Applied transformations to handshake
> [INFO] Built kernel for profiling
> [INFO] Ran kernel for profiling
> [INFO] Profiled cf-level
> [INFO] Running smart buffer placement
> [INFO] Placed smart buffers
> [INFO] Canonicalized handshake
> [INFO] Created visual DOT
> [INFO] Converted visual DOT to PNG
> [INFO] Created loop_accumulate DOT
> [INFO] Converted loop_accumulate DOT to PNG
> [INFO] Compilation succeeded
> ```

Intermediate and final results of `compile` are stored in `tutorials/Introduction/Ch1/out/comp/`.

> [!TIP]
> The `compile` command creates a static visual representation of the generated dataflow circuit in `tutorials/Introduction/Ch1/out/comp/visual.png`. Open it and study the graph corresponding to the dataflow circuit. Nodes in the graph represent dataflow components connected by dataflow channels (edges).

> [!IMPORTANT]
> Now generate the VHDL design using the `write-hdl` command.
> ```
> dynamatic> write-hdl
> [INFO] Converted DOT to VHDL
> [INFO] HDL generation succeeded
> ```

Intermediate and final results of `write-hdl` are stored in `tutorials/Introduction/Ch1/out/hdl/`.

> [!IMPORTANT]
> The VHDL design can finally be co-simulated along the C function on Modelsim to verify that their behavior matches using the `simulate` command.
> ```
> dynamatic> simulate
> [INFO] Built kernel for IO gen.
> [INFO] Ran kernel for IO gen.
> [INFO] Launching Modelsim simulation
> [INFO] Simulation succeeded
> ```

Intermediate and final results of `simulate` are stored in `tutorials/Introduction/Ch1/out/sim/`.

That's it, you have successfully synthesized your first dataflow circuit from C code and functionnaly verified it using Dynamatic!

> [!IMPORTANT]
> At this point, you can quit the Dynamatic frontend by inputting the `exit` command. 
> ```
> dynamatic> exit
> 
> Goodbye!
> ```

### Non-interactive mode

Note that the frontend can be used non-interactively by writing the list of commands one wishes to execute in a text file. You can replay the entire previous subsection at once by running the following from Dynamatic's top-level folder.

```sh
./bin/dynamatic --run tutorials/Introduction/Ch1/frontend-script.sh
```

In the last section of this exercise, we will take a closer look at the actual circuit that was generated by Dynamatic and visualize its execution interactively. 

## Visualizing the resulting dataflow circuit

> [!IMPORTANT]
> To generate the information needed by the visualizer, re-open the frontend, re-set the source, then input the `visualize` command.
> ```
> $ ./bin/dynamatic
> ================================================================================
> ============== Dynamatic | Dynamic High-Level Synthesis Compiler ===============
> ==================== EPFL-LAP - <release> | <release-date> =====================
> ================================================================================
> 
> dynamatic> set-src tutorials/Introduction/Ch1/loop_accumulate.c
> dynamatic> visualize
> [INFO] Generated channel changes
> [INFO] Added positioning info. to DOT
> 
> dynamatic> exit
> 
> Goodbye!
> ```

Intermediate and final results of `visualize` are stored in `tutorials/Introduction/Ch1/out/visual/`.

> [!IMPORTANT]
> Launch the dataflow visualizer from Dynamatic's top-level folder.
> ```sh
> ./bin/visual-dataflow
> ```

On the main menu, the visualizer prompts you to indicate the path to a DOT file and a CSV file to render the visualization.

> [!IMPORTANT]
> Select each file by clicking on each corresponding button in the visualizer's menu and navigating to it in the file explorer that pops up. The DOT and CSV files are located, respetively, at `tutorials/Introduction/Ch1/out/visual/loop_accumulate.dot` and `tutorials/Introduction/Ch1/out/visual/sim.csv`.

You should now see a visual representation of the dataflow circuit you just synthesized. In this example, the original source code had 5 basic blocks, which are transcribed here in 5 labeled rectangular boxes. Two of these basic blocks represent the start and end of the kernel before and after the loop, respectively. The other 3 hold the loop's logic.

> [!TIP]
> Try to identify which basic block is which from the nature of the nodes and from their connections. Consider that the loop may have been slightly transformed by Dynamatic to optimize the resulting circuit.

> [!TIP]
> Observe the circuit executes using the interactive controls at the bottom of the window. On cycle 6, for example, you can see that tokens are transferred on both input channels of `muli0` in `block2`. Try to infer the multiplier's latency by looking at its output channel in the next execution cycles. Then, try to track that output token through the circuit to see where it can end up. Study the execution till you get an understanding of how tokens flow inside the loop and of how the conditional multiplication influences the latency of each loop iteration.

## Conclusion

Congratulations on reaching the end of this exercise! You now know how to use Dynamatic to compile C kernels into functional dataflow circuits, then visualize these circuits to better understand them and identify potential optimization opportunities. In the [next exercise](ModifyingDynamatic.md), we will identify one such opportunity and write a small transformation pass in C++ to implement our desired optimization, before finally verifying its behavior using the dataflow visualizer.
