const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, LevelFormat, PageBreak, VerticalAlign, PageNumber, Header, Footer,
} = require("docx");

const FIG = "/tmp/figures";
const FONT = { ascii: "Calibri", eastAsia: "Calibri", hAnsi: "Calibri", cs: "Calibri" };
const NAVY = "1F3864";
const GREY = "595959";
const INK = "111111";

// ---- helpers -------------------------------------------------------------
function h1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:300,after:120},children:[new TextRun({text:t,bold:true,size:26,color:NAVY,font:FONT})]});}
function h2(t){return new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:200,after:80},children:[new TextRun({text:t,bold:true,size:22,color:NAVY,font:FONT})]});}
function p(t){return new Paragraph({alignment:AlignmentType.LEFT,spacing:{before:60,after:60,line:300,lineRule:"auto"},children:[new TextRun({text:t,size:21,font:FONT,color:INK})]});}
function pm(segs){return new Paragraph({alignment:AlignmentType.LEFT,spacing:{before:60,after:60,line:300,lineRule:"auto"},children:segs.map(([t,b])=>new TextRun({text:t,bold:!!b,size:21,font:FONT,color:INK}))});}
function bullet(t){return new Paragraph({numbering:{reference:"bul",level:0},spacing:{before:40,after:40,line:295,lineRule:"auto"},children:[new TextRun({text:t,size:21,font:FONT,color:INK})]});}
function img(file,w,h){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:40},children:[new ImageRun({type:"png",data:fs.readFileSync(`${FIG}/${file}`),transformation:{width:w,height:h}})]});}
function cap(t){return new Paragraph({alignment:AlignmentType.LEFT,spacing:{before:0,after:150},children:[new TextRun({text:t,size:17,italics:true,color:GREY,font:FONT})]});}
function ti(t){return new Paragraph({spacing:{before:90,after:50},children:[new TextRun({text:t,size:17,italics:true,color:GREY,font:FONT})]});}
function pageBreak(){return new Paragraph({children:[new PageBreak()]});}
function ctr(t,size,bold,color,after){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:after||120},children:[new TextRun({text:t,size:size||22,bold:!!bold,color:color||INK,font:FONT})]});}
function ref(t){return new Paragraph({spacing:{before:40,after:40,line:270,lineRule:"auto"},indent:{left:360,hanging:360},children:[new TextRun({text:t,size:19,font:FONT,color:INK})]});}
function table(rows,widths){const total=widths.reduce((a,b)=>a+b,0);return new Table({width:{size:total,type:WidthType.DXA},columnWidths:widths,rows:rows.map((cells,ri)=>new TableRow({children:cells.map((cc,ci)=>new TableCell({width:{size:widths[ci],type:WidthType.DXA},shading:ri===0?{fill:NAVY,type:ShadingType.CLEAR,color:"auto"}:{fill:ri%2===0?"F2F5FA":"FFFFFF",type:ShadingType.CLEAR,color:"auto"},margins:{top:55,bottom:55,left:110,right:110},verticalAlign:VerticalAlign.CENTER,children:[new Paragraph({spacing:{before:0,after:0},children:[new TextRun({text:cc,size:18,font:FONT,bold:ri===0,color:ri===0?"FFFFFF":"000000"})]})]}))}))});}

const c = [];

// ===================== TITLE PAGE =====================
c.push(new Paragraph({spacing:{before:900,after:0},children:[]}));
c.push(ctr("Imperial College London",26,true,NAVY,40));
c.push(ctr("Department of Earth Science and Engineering",21,false,INK,20));
c.push(ctr("MSc in Applied Computational Science and Engineering",20,false,GREY,320));
c.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:200,after:40},border:{top:{style:BorderStyle.SINGLE,size:6,color:NAVY,space:6}},children:[]}));
c.push(ctr("Microscaling Block Floating-Point Storage for",30,true,NAVY,10));
c.push(ctr("Finite-Difference Seismic Wave Propagation",30,true,NAVY,40));
c.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:20,after:300},border:{bottom:{style:BorderStyle.SINGLE,size:6,color:NAVY,space:6}},children:[]}));
c.push(ctr("Shijun Huang",23,true,INK,300));
c.push(ctr("Individual Research Project",19,false,GREY,10));
c.push(ctr("in collaboration with SLB",19,false,GREY,320));
c.push(ctr("Supervisor: Dr James Hobro (SLB)",19,false,INK,20));
c.push(ctr("Code repository: [to be added]",18,false,GREY,320));
c.push(ctr("August 2026",19,false,INK,10));
c.push(pageBreak());

// ===================== ABSTRACT =====================
c.push(h1("Abstract"));
c.push(p("Finite-difference seismic solvers are limited by memory bandwidth rather than arithmetic, so the number of bits used to store the wavefield sets how fast they run. This project asks whether microscaling, a block floating-point storage format from machine learning in which a group of values shares one exponent, carries the same benefit to seismic modelling. It was implemented as a storage layer around a full-precision finite-difference solver on Marmousi, with the wavefield packed to microscaling and unpacked each step while the arithmetic stayed in single precision. Accuracy was the signal-to-noise ratio of the shot record against a full-precision reference, time-scaled to weight the arrivals evenly, and efficiency was estimated from memory traffic. At about fourteen bits per value microscaling reached a higher signal-to-noise ratio than half precision at sixteen bits, on both acoustic and elastic kernels, and held when only the reflections were scored. Further tests showed the accuracy limit comes from error accumulation over time rather than from the format, and that a square block beats the memory-ordered strip at equal cost. Microscaling storage is thus a practical route to lower memory traffic in wave modelling, with larger gains once the model arrays are compressed too."));

// ===================== 1 INTRODUCTION =====================
c.push(h1("1  Introduction"));
c.push(p("Seismic imaging and full-waveform inversion rest on repeatedly solving the wave equation across large earth models (Virieux and Operto, 2009), and the finite-difference method remains the workhorse for that solve. Its cost is dominated not by floating-point work but by data movement: the wavefield is written to and read from memory at every time step, and on modern hardware the arithmetic units sit idle waiting for that traffic. A solver of this kind is memory-bandwidth-bound, and its speed is governed by how many bytes cross the memory bus, not by how many operations run."));
c.push(p("The direct consequence is that the storage precision of the wavefield is a lever on performance. Every bit removed from a stored value is a byte fraction that need not be moved, so a narrower representation translates into a faster solve, provided the loss of precision does not spoil the result. Reduced precision, and data compression more generally, has therefore drawn steady interest as a way to cut the cost of the solve, most visibly through half precision, which halves storage against single precision and has been demonstrated on the elastic wave equation."));
c.push(p("Machine learning has pursued the same lever further. Block floating-point formats, and in particular microscaling, let a block of values share a single exponent, spending bits on mantissa where flexibility is not needed and recovering the exponent overhead across the block. These formats now underpin low-precision training and inference at scale. Whether the idea transfers to seismic wave propagation, where the stored field spans a wide dynamic range and errors accumulate over hundreds of time steps, is the question this project addresses."));
c.push(p("This report implements microscaling as a storage layer around a finite-difference solver, measures its accuracy and its predicted efficiency against half precision and other sixteen-bit formats, and explains the mechanism that governs the result. The work is deliberately confined to two dimensions and to an emulated format, so that the accuracy findings, which are independent of hardware, can be established cleanly before an optimised kernel is written."));

// ===================== 2 BACKGROUND =====================
c.push(h1("2  Background and contribution"));
c.push(p("Reduced-precision seismic modelling was placed on a firm footing by Fabien-Ouellet (2020), who solved the isotropic elastic wave equation in half precision. Half precision has a narrow exponent range, so a naive solve overflows or underflows; the method applies a non-linear, logarithmic scaling of the wave equation, with the scaling factor derived from the model parameters, to keep the computation in range. The arithmetic itself runs in sixteen bits, which reduces both compute and storage, and on a graphics processor the approach was roughly twice as fast as single precision and used half the memory."));
c.push(p("The format studied here differs in what it narrows and in how the scale is chosen. Microscaling is a storage precision: the arithmetic stays in single precision, and only the wavefield held in memory between steps is compressed, so the saving is in memory traffic rather than in compute. The scale is not a single global factor chosen once from the model, but one exponent per small block, set automatically from that block's own values. Where a global scale must accommodate the whole field at once, a per-block scale adapts to each region, which matters for a wavefield whose amplitudes vary over several orders of magnitude between the strong direct arrival and the weak reflections."));
c.push(p("General-purpose floating-point compressors such as ZFP (Lindstrom, 2014) also reduce the footprint of scientific arrays, and remain a natural baseline; the contribution here is narrower and more specific. It is to establish, on a standard seismic benchmark and against the half-precision baseline in current use, that a per-block shared exponent reaches better accuracy per stored bit than either half precision or a single global fixed-point scale, and to identify why. To our knowledge microscaling block floating-point has not previously been applied to finite-difference wave propagation, and the block-shape effect reported in Section 5.6 has not been described in this setting."));

// ===================== 3 OBJECTIVES =====================
c.push(h1("3  Objectives"));
c.push(p("The project set out to answer four measurable questions:"));
c.push(bullet("Whether microscaling reaches a higher signal-to-noise ratio than half precision at equal or lower bits per stored value, on the acoustic and elastic kernels of a standard model."));
c.push(bullet("Whether that advantage survives a stricter, reflection-only accuracy measure that excludes the strong direct arrival."));
c.push(bullet("What sets the accuracy limit, distinguishing the error made by the format on a single frame from the error accumulated over a full solve."));
c.push(bullet("How the shape of the block, at fixed size and bit cost, affects accuracy, and whether it affects the cost of packing and unpacking."));

// ===================== 4 METHODS =====================
c.push(h1("4  Methods"));

c.push(h2("4.1  Storage format"));
c.push(p("A value is stored by grouping the wavefield into blocks of thirty-two and giving each block one shared exponent while each value keeps its own mantissa. The shared factor is a power of two, held as an eight-bit exponent and set from the largest magnitude in the block, so applying and removing it introduces no rounding of its own. Each value then carries a sign bit, an implicit integer bit, and a mantissa whose width is the parameter swept in the experiments. The cost per value is the mantissa width plus two, plus the block's share of the exponent, eight divided by the block size; at twelve mantissa bits and a block of thirty-two this is 14.25 bits, against the flat sixteen of half precision. The implementation is a simplified block floating-point rather than the exact OCP microscaling standard, which assigns each element its own micro-exponent; the distinction is returned to in the discussion."));
c.push(p("The separation between storage and computation precision is central and holds throughout. Microscaling narrows storage while the finite-difference arithmetic runs in single precision; half precision as used by Fabien-Ouellet narrows the arithmetic itself. The two save different resources and are not interchangeable, and unless a comparison explicitly reproduces half-precision arithmetic, every scheme in this report is evaluated purely as a storage format with the arithmetic held fixed."));

c.push(h2("4.2  Model and solver"));
c.push(p("Experiments use the Marmousi velocity model (Versteeg, 1994) with a single source and a line of receivers along the surface. The solver is generated by Devito (Louboutin et al., 2019), which compiles a finite-difference operator from a symbolic statement of the wave equation. At each time step the operator advances the field in single precision, after which the wavefield is packed to microscaling and unpacked before the next step, so the format acts only on the stored state between steps. Devito cannot construct a half-precision grid, so the half-precision-arithmetic method of Fabien-Ouellet is reproduced separately in a self-contained stepper (Appendix C). Accuracy sweeps use a cropped window of the model, which shortens each run and allows the many repetitions the sweeps require."));

c.push(h2("4.3  Accuracy and efficiency metrics"));
c.push(p("Accuracy is the relative error of the shot record against a full-precision run through the same driver, reported as a signal-to-noise ratio in decibels, minus twenty times the base-ten logarithm of that relative error. Higher is better and a tenfold reduction in error is twenty decibels. A plain error of this kind is dominated by the strong early arrivals, because they carry most of the record's energy, so it mainly measures how well the top of the record is preserved and barely reflects the weak late arrivals that make up the image. To weight the arrivals evenly, both records are multiplied by a gain that grows linearly with time, the standard correction for geometric spreading, before the error is formed; this lifts the weak deep events to a comparable weight so the metric scores the whole record rather than favouring its strongest part. Every signal-to-noise figure quoted below is this time-scaled measure."));
c.push(p("A reference point frames the scale: single precision itself departs from double precision by about 9x10^-4 on the full model. Section 5.5 gives a second, independent check on the same concern, removing the direct arrival physically with a water model rather than down-weighting it."));;
c.push(p("Efficiency is estimated from memory traffic rather than measured on the clock. The format is emulated, so timing would record the emulation and not the format, whereas the byte count is exact. Because the solver is bandwidth-bound the predicted speed-up is essentially the ratio of bytes moved, which is the quantity the model computes and which does not depend on the processor."));
c.push(p("With round-to-nearest the accuracy measurements are deterministic: repeating a run reproduces the reported error exactly, so no run-to-run error bars apply. The one measured quantity that does vary between runs is the wall-clock time of packing and unpacking in Section 5.6, which is reported as the fastest of several repeated batches to suppress scheduling noise."));

// ===================== 5 RESULTS =====================
c.push(h1("5  Results"));

c.push(h2("5.1  Accuracy on the acoustic kernel"));
c.push(img("C1_cost_vs_accuracy.png",410,270));
c.push(cap("Figure 1. Cost against accuracy for every scheme on the cropped Marmousi acoustic model. Lower and further left is better; the right axis is the time-scaled signal-to-noise ratio in decibels."));
c.push(pm([["Every format was reimplemented in one harness and run on the same model with the same metric, so the comparison does not rest on numbers drawn from different studies. ",false],["Microscaling reaches 4.1x10^-3 error, or 47.8 dB, at 14.25 bits per value, against 8.2x10^-3, or 41.7 dB, for half precision with scaling at sixteen bits (Table 1).",true],[" It is more accurate while storing fewer bits, and by a wider margin than a plain error would suggest, because the time-scaled metric rewards the weak late arrivals that microscaling preserves better than the alternatives. The same holds against bfloat16, whose wide exponent range is wasted on a field that needs precision, not range, and against int16, whose single global fixed-point scale does worse still, for reasons taken up in the discussion.",false]]));
c.push(ti("Table 1. Accuracy and cost on the acoustic kernel; the time-scaled signal-to-noise ratio is the primary column."));
c.push(table([["Scheme","Bits/value","SNR (dB)","Rel. error"],["FP32 reference","32.00","87.7","4.1x10^-5"],["FP16 + scaling","16.00","41.7","8.2x10^-3"],["int16 + scaling","16.00","37.2","1.4x10^-2"],["bfloat16","16.00","21.6","8.3x10^-2"],["MX, 12 mantissa bits","14.25","47.8","4.1x10^-3"],["MX, 14 mantissa bits","16.25","56.1","1.6x10^-3"]],[2600,1500,1300,1600]));
c.push(img("C3_trace_errors.png",430,237));
c.push(img("H2_water_gather.png",430,235));
c.push(cap("Figure 2. Time-scaled differences, two ways. (a) Each scheme's receiver gather minus the full-precision reference, on one common scale, so a larger residual is a worse scheme: bfloat16 leaves the most, microscaling the least. (b) The full record, the water model that carries only the direct arrival, and their difference, which is the reflections; the water subtraction of Section 5.5 isolates the same weak energy that the time gain in (a) brings out."));
c.push(p("Plotting the gathers directly is uninformative, since at a usable setting microscaling and half precision both reproduce the record closely and the panels look alike, which is itself a statement that both are accurate. Figure 2(a) plots the residual against the reference instead, with a linear time gain applied to lift the weak late arrivals: bfloat16 leaves a large residual across the record and microscaling the smallest, with half precision between. Figure 2(b) shows why the late arrivals matter and how they are isolated, by subtracting a water model that carries only the direct arrival; what remains is the reflected energy, and it is on this energy that the time-scaled metric puts most of its weight."));
c.push(img("B3_wavefield_comparison.png",300,300));
c.push(cap("Figure 3. The reference wavefield and the microscaling wavefield with their difference, at fourteen and ten mantissa bits, each difference shown on its own finer scale."));
c.push(pm([["The closeness is visible in the wavefield itself, not only in the recorded traces. ",false],["At fourteen mantissa bits the microscaling wavefield is indistinguishable from the reference, and the difference has to be drawn on a scale two orders of magnitude finer to show any structure at all; at ten bits the difference is larger but still faint and confined to the strong wavefronts",true],[". This confirms that the format preserves the propagating field, and not merely the receiver record, which is what makes it usable as an in-place storage layer.",false]]));

c.push(h2("5.2  Predicted efficiency"));
c.push(img("D1_efficiency_vs_error.png",400,255));
c.push(cap("Figure 4. Predicted speed-up against error, for the whole solver's memory traffic (all arrays, system level). The dotted line marks the ceiling reachable if the static model arrays were also compressed."));
c.push(pm([["At matched accuracy the bandwidth model gives microscaling a predicted speed-up of about 1.6 times, against 1.5 for half precision. ",false],["The gain is capped near twofold because only the wavefield is compressed while the model arrays remain in single precision",true],[", and that cap, not the format, is what limits the system-level result. Compressing the static model arrays, a low-risk change since they do not accumulate error, would raise the ceiling towards fivefold and is the most valuable next step.",false]]));

c.push(h2("5.3  The elastic kernel"));
c.push(img("E1_elastic_comparison.png",405,262));
c.push(cap("Figure 5. Elastic kernel, per-field storage accuracy: microscaling against a per-field half-precision scaling, same model and metric. The right axis is the time-scaled signal-to-noise ratio in decibels."));
c.push(pm([["On the elastic kernel, the setting half-precision scaling was designed for, microscaling again reaches better accuracy at fewer bits. ",false],["At 14.25 bits it gives 47 dB against 41 dB for the per-field scaled half precision at sixteen bits, and even at eleven mantissa bits, 13.25 bits per value, it holds 44 dB, above the half-precision point while spending nearly three bits per value less.",true],[" The crossover where microscaling matches the baseline lies near ten to eleven mantissa bits. One qualification is due: the per-field scaling improves on naive half precision by only about four percent here, because the stored fields span only a narrow range, so this is a weak baseline, and the fair claim is the narrower one that a per-block exponent matches or beats a per-field global scale at fewer bits.",false]]));

c.push(h2("5.4  What limits the accuracy"));
c.push(img("G1_compression_only.png",380,240));
c.push(cap("Figure 6. Single-frame error, from compressing and decompressing once, against the error accumulated over the full solve, in decibels. The gap is the amplification propagation adds."));
c.push(pm([["The accuracy limit is not set by the format's error on any one frame. ",true],["A single compression and decompression at twelve mantissa bits loses only about 78 dB, effectively nothing, yet the same format run through the solve sits near 35 dB on the wavefield: propagation amplifies the single-frame error by roughly one to two hundred times (Figure 6).",false],[" Because the wavefield is read and written every step, each rounding enters the next step and is amplified, so the binding constraint is accumulation over time, not representation. The practical corollary is that adding mantissa bits attacks the wrong term; suppressing accumulation, for instance by quantising less often, is the more efficient route and is left for further work.",false]]));

c.push(h2("5.5  A second check: scoring the reflections directly"));
c.push(img("H1_water_snr.png",380,240));
c.push(cap("Figure 7. Time-scaled whole-record against reflection-only signal-to-noise ratio by bit width. The reflections are isolated by subtracting a water model that carries only the direct arrival (Figure 2b)."));
c.push(p("The time-scaled metric down-weights the strong direct arrival, but it does not remove it, so it is worth confirming the result with a method that removes it outright. A second model that is water everywhere at the surface velocity was run with the source, geometry and time-stepping held identical and only the velocity changed. With no contrasts it produces the direct arrival alone, and subtracting it isolates the reflections; the microscaling run is put through the water model too, so the arrival removed is the one it produced."));
c.push(pm([["The two routes agree. ",true],["On this model the reflections carry about 45 percent of the record's energy, and scored on them alone microscaling reaches 43 dB at twelve mantissa bits, close to its 48 dB time-scaled whole-record figure and still clear of a 40 dB target. The gap between the two measures is about five decibels and holds across bit widths, and the ranking of the schemes does not change",true],[". Removing the direct arrival physically and down-weighting it by time-scaling give the same verdict, which is the point: microscaling's advantage does not depend on the strong arrival that any format reproduces easily.",false]]));

c.push(h2("5.6  Block shape"));
c.push(img("I1_block_shape.png",430,180));
c.push(cap("Figure 8. Accuracy (left) and pack-and-unpack time (right) against block shape, at fixed size and bit cost. Aspect ratio one is a square block, higher is more strip-like."));
c.push(pm([["A block of thirty-two values was found to be more accurate than smaller blocks, which is the wrong way round if size alone mattered, and the cause is shape. ",true],["Holding the block size and bit cost fixed and varying only the shape, the memory-ordered strip gives 23.9 dB and a square block 28.5 dB, a gain of 4.6 dB at the same cost (Figure 8).",false],[" A compact block draws its values from a small region where amplitudes are alike, so the shared exponent fits them better than it does across a strip that runs down the field through different arrivals. The square block needs a two-dimensional tiling to pack and unpack and so costs about 1.1 to 1.2 times the strip in that step, but the storage cost, and therefore the predicted speed-up, is identical, and in a bandwidth-bound solver the extra arithmetic is negligible. The lesson is that blocks should follow the geometry of the field, which points to cube-shaped blocks in three dimensions.",false]]));

// ===================== 6 DISCUSSION =====================
c.push(h1("6  Discussion"));
c.push(p("The results are consistent in showing that a per-block shared exponent uses each stored bit more effectively than the alternatives, and the comparison with int16 explains why. On paper int16, with fifteen mantissa bits, should beat the ten of half precision, yet it does not, because a single global fixed-point scale cannot span the wavefield's dynamic range: with the strong direct arrival a thousand times the weak reflections, a typical value keeps only about three bits of real precision, where floating point keeps ten everywhere. Restoring a scale per block lifts int16 above half precision and, as the block shrinks, into the microscaling range. int16 is thus the one-block limit of the same family, and microscaling is int16 given the local scales it was missing. Bit allocation, not bit count, is what determines accuracy, which is also why bfloat16, rich in range but poor in mantissa, does worst of all."));
c.push(pm([["The limitations fall into two kinds. ",true],["One is inherent to the method. The system-level speed-up is capped near twofold because only the wavefield is compressed while the model arrays are not, and error accumulation over time steps, shown in Section 5.4 to be the binding constraint, is a property of feeding a quantised field back into the solve rather than of the present implementation. The other kind is a limitation of this study, and could be lifted with more time. Efficiency is modelled rather than measured because the format is emulated; the quantiser is a simplified block floating-point rather than the exact OCP standard, so published guidance on safe bit widths would require that standard; and the experiments are two-dimensional and single-shot on cropped models. Two smaller questions were left open by the study: stochastic rounding did not improve on round-to-nearest, contrary to results for the parabolic heat equation (Croci and Giles, 2023), and the effect of quantising less frequently was not swept. The block-size anomaly, by contrast, is now explained as a shape effect.",false]]));

// ===================== 7 CONCLUSIONS =====================
c.push(h1("7  Conclusions and future work"));
c.push(p("The four objectives set out in Section 3 were all met. On the first, microscaling reached a higher signal-to-noise ratio than half precision at fewer bits per stored value, on both kernels: 47.8 dB against 41.7 dB at 14.25 bits on the acoustic kernel, and 46.7 dB, or 43.6 dB at only 13.25 bits, against 40.6 dB on the elastic kernel. On the second, that advantage survived a physical removal of the direct arrival: scored on the reflections alone microscaling reached about 43 dB at twelve mantissa bits, close to its whole-record figure and still clear of a 40 dB target, with the ranking of the schemes unchanged. On the third, the accuracy limit was shown to come from error accumulation over time steps rather than from the format's single-frame error, which a single compress-and-decompress preserves to about 78 dB while the full solve amplifies it a hundredfold or more. On the fourth, block shape was found to affect accuracy at fixed cost, a square block beating the memory-ordered strip by 4.6 dB, while adding only a 1.1 to 1.2 times pack-and-unpack overhead that is negligible in a bandwidth-bound solver."));;
c.push(p("Taken together these establish microscaling storage as a practical means of reducing the memory traffic that governs solver speed, more accurate per stored bit than the half-precision baseline in current use, and understood well enough to say where its limit lies and how its blocks should be arranged."));;
c.push(p("The immediate next step is to compress the static model arrays, which the efficiency model identifies as the largest available gain, and to move from the simplified quantiser to the exact OCP microscaling standard so that safe bit widths map to a published format. The block-shape result points to cube-shaped blocks in three dimensions, where the case is expected to strengthen: three-dimensional solvers are more strongly bandwidth- and capacity-bound, and large models often do not fit in device memory, so a format that moves fewer bytes helps more there, not less."));
c.push(p("Realising the predicted speed-up on hardware is the natural continuation, and the distinction between the present work and a hardware implementation is worth stating plainly. Here the format is emulated: the accuracy is exact and independent of the processor, while the efficiency is derived from memory traffic. A graphics-processor implementation would move the packing and unpacking into the compute kernel, so the wavefield genuinely occupies fewer bytes in device memory, and the speed-up would then be timed rather than modelled. The accuracy results carry over unchanged, since precision loss does not depend on hardware, and the bandwidth argument only strengthens on a processor that is more bandwidth-bound than the one used here. What a hardware run would add is the one thing the model cannot see, namely whether the packing and unpacking, once in a real kernel, stay cheap enough to preserve the saving; the timings measured here suggest they do."));

// ===================== ACKNOWLEDGEMENTS =====================
c.push(h1("Acknowledgements"));
c.push(p("I thank Dr James Hobro of SLB for supervising this project and for guidance throughout, and SLB for proposing and sponsoring the work."));
c.push(p("Generative AI (Anthropic's Claude) was used as an assistant during the project. Its use covered drafting and refining code for the experiments, help with debugging, exploratory analysis of results, and language editing of this report. All experimental design, the interpretation of results, and the final content were reviewed and verified by the author, who takes full responsibility for the work."));

// ===================== REFERENCES =====================
c.push(h1("References"));
c.push(ref("Croci, M., and Giles, M. B. (2023). Effects of round-to-nearest and stochastic rounding in the numerical solution of the heat equation in low precision. IMA Journal of Numerical Analysis, 43(3), 1358-1390."));
c.push(ref("Fabien-Ouellet, G. (2020). Seismic modeling and inversion using half-precision floating-point numbers. Geophysics, 85(3), F65-F76."));
c.push(ref("Lindstrom, P. (2014). Fixed-rate compressed floating-point arrays. IEEE Transactions on Visualization and Computer Graphics, 20(12), 2674-2683."));
c.push(ref("Louboutin, M., Lange, M., Luporini, F., Kukreja, N., Witte, P. A., Herrmann, F. J., Velesko, P., and Gorman, G. J. (2019). Devito (v3.1.0): an embedded domain-specific language for finite differences and geophysical exploration. Geoscientific Model Development, 12(3), 1165-1187."));
c.push(ref("Open Compute Project (2023). OCP Microscaling Formats (MX) Specification, Version 1.0."));
c.push(ref("Versteeg, R. (1994). The Marmousi experience: velocity model determination on a synthetic complex data set. The Leading Edge, 13(9), 927-936."));
c.push(ref("Virieux, J., and Operto, S. (2009). An overview of full-waveform inversion in exploration geophysics. Geophysics, 74(6), WCC1-WCC26."));

// ===================== APPENDICES =====================
c.push(pageBreak());
c.push(h1("Appendix A  Baseline validation"));
c.push(p("The full-precision baseline against which every scheme is measured. Figure A1 shows the velocity model and acquisition; Figure A2 gives the single- against double-precision difference that fixes the accuracy reference at about 9x10^-4 on the full model."));
c.push(img("A1_velocity_model.png",360,150));
c.push(cap("Figure A1. Marmousi velocity model with source and receiver positions."));
c.push(img("A4_fp32_vs_fp64.png",330,205));
c.push(cap("Figure A2. Single- against double-precision difference, which sets the accuracy reference."));

c.push(h1("Appendix B  Supporting figures"));
c.push(p("Figure B1 shows how the wavefield error builds up over the solve, the growth behind the accumulation argument of Section 5.4. Figure B2 is the int16 diagnostic of Section 6, where giving int16 a scale per block drops its error below half precision and converges towards microscaling as the block shrinks."));
c.push(img("B2_error_growth.png",360,225));
c.push(cap("Figure B1. Growth of the relative wavefield error over the solve at six mantissa bits, on a logarithmic axis."));
c.push(img("F1_int16_improvement.png",360,229));
c.push(cap("Figure B2. int16 error as the number of independent scales increases; per-block scaling converges to microscaling."));

c.push(h1("Appendix C  Half-precision arithmetic"));
c.push(p("Because Devito cannot build a half-precision grid, the half-precision-arithmetic method of Fabien-Ouellet was reproduced in a self-contained stepper (Figure C1), which reproduces the reported behaviour and confirms the baseline the elastic comparison scores against. Table C1 gives the elastic-kernel figures in full."));
c.push(img("E2_fabien_reproduction.png",330,210));
c.push(cap("Figure C1. Reproduction of half-precision arithmetic in a self-contained stepper."));
c.push(ti("Table C1. Accuracy and cost on the elastic kernel."));
c.push(table([["Scheme","Bits/value","SNR (dB)","Rel. error"],["FP16, no scaling","16.00","40.2","9.7x10^-3"],["FP16, per-field scaling","16.00","40.6","9.3x10^-3"],["MX, 10 mantissa bits","12.25","40.1","9.9x10^-3"],["MX, 11 mantissa bits","13.25","43.6","6.6x10^-3"],["MX, 12 mantissa bits","14.25","46.7","4.6x10^-3"],["MX, 14 mantissa bits","16.25","63.0","7.1x10^-4"]],[2900,1400,1300,1600]));

// ---- numbering + build ----
const doc = new Document({
  features:{updateFields:true},
  numbering:{config:[{reference:"bul",levels:[{level:0,format:LevelFormat.BULLET,text:"\u2022",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:360,hanging:220}}}}]}]},
  styles:{default:{document:{run:{font:FONT,size:21,color:INK}}}},
  sections:[{
    properties:{page:{margin:{top:1200,right:1200,bottom:1200,left:1200}},titlePage:true},
    footers:{
      first:new Footer({children:[new Paragraph({children:[]})]}),
      default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],size:18,color:GREY,font:FONT})]})]}),
    },
    children:c,
  }],
});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("final_report.docx",b);console.log("written",b.length);});
