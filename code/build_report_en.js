const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, LevelFormat, PageBreak, VerticalAlign,
} = require("docx");

const FIG = "/tmp/figures";
const FONT = { ascii: "Calibri", eastAsia: "Calibri", hAnsi: "Calibri", cs: "Calibri" };
const NAVY = "1F3864";
const GREY = "595959";

function h1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,spacing:{before:320,after:140},children:[new TextRun({text:t,bold:true,size:28,color:NAVY,font:FONT})]});}
function h2(t){return new Paragraph({heading:HeadingLevel.HEADING_2,spacing:{before:240,after:90},children:[new TextRun({text:t,bold:true,size:23,color:NAVY,font:FONT})]});}
function p(t){return new Paragraph({spacing:{before:60,after:60,line:290,lineRule:"auto"},children:[new TextRun({text:t,size:21,font:FONT})]});}
function pm(segs){return new Paragraph({spacing:{before:60,after:60,line:290,lineRule:"auto"},children:segs.map(([t,b])=>new TextRun({text:t,bold:!!b,size:21,font:FONT}))});}
function bulletm(segs){return new Paragraph({numbering:{reference:"bul",level:0},spacing:{before:50,after:50,line:285,lineRule:"auto"},children:segs.map(([t,b])=>new TextRun({text:t,bold:!!b,size:21,font:FONT}))});}
function img(file,w,h){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:50},children:[new ImageRun({type:"png",data:fs.readFileSync(`${FIG}/${file}`),transformation:{width:w,height:h}})]});}
function cap(t){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:0,after:150},children:[new TextRun({text:t,size:17,italics:true,color:GREY,font:FONT})]});}
function pageBreak(){return new Paragraph({children:[new PageBreak()]});}
function ti(t){return new Paragraph({spacing:{before:100,after:60},children:[new TextRun({text:t,size:18,italics:true,color:GREY,font:FONT})]});}
function table(rows,widths){const total=widths.reduce((a,b)=>a+b,0);return new Table({width:{size:total,type:WidthType.DXA},columnWidths:widths,rows:rows.map((cells,ri)=>new TableRow({children:cells.map((cc,ci)=>new TableCell({width:{size:widths[ci],type:WidthType.DXA},shading:ri===0?{fill:NAVY,type:ShadingType.CLEAR,color:"auto"}:{fill:ri%2===0?"F2F5FA":"FFFFFF",type:ShadingType.CLEAR,color:"auto"},margins:{top:60,bottom:60,left:110,right:110},verticalAlign:VerticalAlign.CENTER,children:[new Paragraph({spacing:{before:0,after:0},children:[new TextRun({text:cc,size:18,font:FONT,bold:ri===0,color:ri===0?"FFFFFF":"000000"})]})]}))}))});}

const c=[];

c.push(new Paragraph({spacing:{before:0,after:60},children:[new TextRun({text:"Microscaling Storage Formats for Seismic Wave Propagation",bold:true,size:36,color:NAVY,font:FONT})]}));
c.push(new Paragraph({spacing:{before:0,after:40},children:[new TextRun({text:"Interim experimental report: results, strengths and limitations",size:23,color:GREY,font:FONT})]}));
c.push(new Paragraph({spacing:{before:0,after:200},border:{bottom:{style:BorderStyle.SINGLE,size:8,color:NAVY,space:4}},children:[new TextRun({text:"SLB and Imperial College London industry sponsored research project (IRP)    Shijun Huang",size:19,color:GREY,font:FONT})]}));

// Summary
c.push(h1("Summary"));
c.push(pm([["Microscaling (MX) block formats, developed for machine learning, have been applied to a finite difference seismic wave solver and compared against the established reduced precision schemes on the same model. The findings: ",false],["as a storage format MX is cheaper and more accurate than FP16",true],[", on the acoustic and elastic kernels; ",false],["compressing only the wavefield caps the speedup at about 2x",true],[" until the model arrays are compressed too; a full reproduction of Fabien-Ouellet's FP16 arithmetic method shows that ",false],["FP16 and MX are complementary, not rivals",true],["; and ",false],["combining them does not compound the error",true],[", so FP16 arithmetic and sub-16-bit MX storage can be used together.",false]]));
c.push(p("All figures below are from runs carried out for this report. No numbers are estimated."));

// Setup
c.push(h1("Experimental setup"));
c.push(bulletm([["Model. ",true],["Marmousi, the standard synthetic velocity model in seismic imaging, 7.5 km wide and 3 km deep, with faults and dipping layers. Accuracy sweeps use a window cut from it, since each sweep requires many repeated runs."]]));
c.push(bulletm([["How MX is applied. ",true],["At every time step the Devito kernel advances the wave in FP32. The wavefield is packed into MX, unpacked back to FP32, and fed into the next step. Arithmetic stays at native precision; only the storage is narrowed."]]));
c.push(bulletm([["Accuracy metric. ",true],["Relative L2 error of the shot record against a full precision run through the same driver, reported primarily as signal-to-noise ratio in decibels, SNR(dB) = -20 log10 of that relative error. Higher decibels is better, and a factor of ten in error is twenty decibels. One caveat, carried through the report: this global measure is dominated by the strong direct arrival, so it mainly reflects how well the strong part of the record is preserved; a reflection-focused metric is planned."]]));
c.push(bulletm([["Cost metric. ",true],["Bits stored per wavefield value. The solver is memory bandwidth bound, so fewer bits stored means fewer bytes moved each step."]]));
c.push(p("One reference number frames everything. FP32 already differs from FP64 by 9.0e-4 on the full Marmousi model, and by 2.6e-5 on the cropped run. That is the bar: an MX setting whose error reaches it is as accurate as FP32 in practice."));
c.push(pm([["A note on scope. ",true],["Fabien-Ouellet (2020) developed FP16 seismic modelling on the isotropic elastic equation, scaling the arithmetic to keep computation in range. Most of this report compares storage formats with arithmetic held at FP32; the FP16 with scaling baseline is a storage side reproduction in that spirit. The final two sections then reproduce his arithmetic method properly and test combining it with MX.",false]]));

c.push(pageBreak());

// Storage format specification
c.push(h1("Storage format specification"));
c.push(p("This section states exactly how a value is stored, so the bit costs quoted elsewhere are unambiguous and the format can be reimplemented from the description alone."));
c.push(h2("Block structure"));
c.push(pm([["The wavefield array is flattened and cut into blocks of ",false],["32 consecutive values",true],[". All values in a block share one exponent; each value keeps its own mantissa. The block is a one-dimensional run along the array's fastest storage axis, not a square or cube in physical space; the effect of that choice is examined in the block size discussion, and revisiting it is listed as future work.",false]]));
c.push(h2("Bit allocation per value"));
c.push(bulletm([["Sign: 1 bit",true],[", per value."]]));
c.push(bulletm([["Mantissa: the swept parameter",true],[", from 4 to 14 bits per value, plus one implicit integer bit. The headline setting is 12 mantissa bits."]]));
c.push(bulletm([["Shared exponent: 8 bits per block",true],[", so 8 divided by the block size of 32, which is 0.25 bits, charged to each value."]]));
c.push(pm([["The total is ",false],["mantissa + 2 + 8/blocksize",true],[" bits per value: the mantissa, a sign and an integer bit, and the block's share of the exponent. At 12 mantissa bits and block 32 this is 12 + 2 + 0.25 = 14.25 bits per value, against FP16's flat 16.",false]]));
c.push(h2("Scaling-factor representation"));
c.push(pm([["The shared factor is a ",false],["power of two, stored as an 8-bit exponent",true],[", set from the block's largest magnitude: the exponent is the floor of the base-two log of that maximum. Each value in the block is then divided by that power of two and rounded to the mantissa grid. Because the factor is a power of two, applying it and removing it is exact and adds no rounding of its own. Decoding is the reverse: read the mantissa, multiply by the shared power of two.",false]]));
c.push(h2("Storage precision against computation precision"));
c.push(pm([["This is the distinction that separates the schemes in this report and is stated here once. ",true],["MX is a storage precision: the finite-difference arithmetic runs in FP32, and only the wavefield in memory between steps is narrowed. FP16 as used by Fabien-Ouellet is a computation precision: the arithmetic itself runs in 16 bits. The two save different resources, storage bandwidth versus compute, and are not interchangeable. Throughout, unless a section explicitly reproduces FP16 arithmetic, every scheme is compared as a storage format with FP32 arithmetic held fixed, so the comparison isolates the storage format alone.",false]]));

c.push(pageBreak());

// Acoustic results
c.push(h1("Principal results, acoustic kernel"));
c.push(h2("Cost against accuracy"));
c.push(img("C1_cost_vs_accuracy.png",470,305));
c.push(cap("Figure 1. All schemes on the same cropped Marmousi acoustic model. Lower and further left is better; the right axis reads the same thing as signal-to-noise ratio in decibels."));
c.push(p("Cost runs along the horizontal axis and error up the vertical, so a scheme is better the closer it sits to the bottom left corner. The right axis carries the same information as signal-to-noise ratio in decibels, which is how the field usually reports it: SNR in dB is minus twenty times the log of the relative error, so every tenfold drop in error is twenty decibels, and higher is better."));
c.push(pm([["The MX curve passes below and to the left of the FP16 point. MX reaches 3.37e-3 at 14.25 bits per value, while FP16 spends 16 bits and reaches only 6.13e-3. ",false],["MX is cheaper and more accurate at once",true],[". A per block shared exponent, which tracks the local amplitude of the wavefield, outperforms a single fixed global scale.",false]]));
c.push(pm([["bfloat16 supports the same conclusion from the opposite direction: at the same 16 bits its error is an order of magnitude worse, because it spends its bits on an exponent range the acoustic wavefield never uses. ",false],["How the exponent budget is allocated is what matters",true],[", not the total width alone. In this acoustic setting the per field scaling has almost nothing to correct; its role appears only on the elastic kernel below.",false]]));
c.push(pm([["int16 with scaling sits at 8.09e-3, above the FP16 formats. It is fixed point with one scale for the whole grid, so a value far below the field maximum keeps almost none of its bits. It is worth including because it is the large block limit of microscaling: ",false],["one shared scale for the entire grid is exactly MX with the block size grown to the whole field",true],[". That it trails MX at block 32 is the same effect the block size study shows from the other end, and it frames the schemes as one family. As the block shrinks from the whole grid (int16) towards 32 (MX), the shared exponent tracks the local amplitude more closely and the accuracy improves.",false]]));
c.push(ti("Table 1. Measured accuracy and cost, acoustic kernel. SNR in decibels is the primary accuracy column; relative error is the same quantity in the other unit."));
c.push(table([["Scheme","Bits per value","SNR (dB)","Relative error","Compression"],["FP32 (baseline)","32.00","91.6","2.64e-05","1.00x"],["FP16, no scaling","16.00","44.9","5.70e-03","2.00x"],["FP16 + scaling","16.00","44.2","6.13e-03","2.00x"],["int16 + scaling","16.00","41.8","8.09e-03","2.00x"],["bfloat16","16.00","24.2","6.13e-02","2.00x"],["MX, 12 mantissa bits","14.25","49.5","3.37e-03","2.25x"],["MX, 14 mantissa bits","16.25","59.8","1.02e-03","1.97x"]],[2500,1500,1300,1700,1300]));
c.push(pm([["In decibels the headline is that ",false],["MX gives about 49 dB at roughly fourteen bits, some five decibels above FP16 at sixteen",true],[", with the FP32-against-FP64 pass mark at about 92 dB on this cropped run. Five decibels is a little under a factor of two in error, gained while spending fewer bits.",false]]));

c.push(pageBreak());
c.push(h2("Efficiency against error"));
c.push(img("D1_efficiency_vs_error.png",470,303));
c.push(cap("Figure 2. Predicted speedup against error for the acoustic kernel, for the whole solver's memory traffic (system level, all arrays). The dotted line is the ceiling reachable if the model arrays were compressed as well."));
c.push(p("Runtime is not timed here, and could not honestly be. The MX layer is emulated in numpy, so packing and unpacking adds work rather than removing it, and Devito's generated kernel continues to read and write FP32 whatever is done to the array between steps. A wall clock figure would measure the emulation, not the format. Efficiency is therefore modelled from memory traffic, which does not depend on the particular GPU available."));
c.push(pm([["At matched accuracy MX is predicted to run at 1.59x against FP16's 1.50x. The dotted line is more significant: it shows what would be reachable if the model arrays were narrowed too, and ",false],["the gap between the two curves is performance currently left unused",true],[".",false]]));
c.push(pm([["The cause is direct. ",false],["Only the wavefield is compressed; the model arrays remain in FP32",true],[" and are moved every step regardless. Even at 6.25 bits per value, a 5.12x compression of the wavefield itself, the acoustic kernel gains only 2.16x overall. This is a hard ceiling.",false]]));

c.push(pageBreak());
// Mechanism
c.push(h1("Mechanism"));
c.push(h2("Error accumulation is the binding constraint"));
c.push(img("B2_error_growth.png",430,269));
c.push(cap("Figure 3. Growth of the relative wavefield error over the simulation, at 6 mantissa bits. Logarithmic vertical axis."));
c.push(pm([["A single pass through MX loses only 0.4 percent. After 770 time steps the error has grown to 129 percent, an amplification of roughly three hundred times. The wavefield is read and written every step, so each quantisation error enters the next step and is amplified. ",false],["The limit is set by this accumulation, not by the representation error of any single frame",true],[".",false]]));
c.push(p("Higher compression will not come from adding mantissa bits, but from suppressing the accumulation: quantising less often, changing the rounding mode, or compressing only a subset of the arrays."));
c.push(h2("Compression on its own, without propagation"));
c.push(p("The claim that accumulation is the constraint can be tested directly by taking propagation out of the loop. A real wavefield is compressed to MX and decompressed once, with no time stepping, and the single-frame error is measured against the same format run through the full solve. If the two differ greatly, the format is not the problem; the propagation is."));
c.push(img("G1_compression_only.png",430,273));
c.push(cap("Figure 4a. Single-frame representation error (compress and decompress once) against the error accumulated over the full solve, in decibels. Red numbers are the amplification propagation adds."));
c.push(pm([["The gap is large and settles the question. ",true],["At 12 mantissa bits a single compress and decompress loses about 78 dB on one frame, effectively lossless, while the same format run through the solve sits near 35 dB. Propagation amplifies the single-frame error by roughly a hundred to two hundred times across the useful bit widths. ",false],["So the bits themselves are not the limit; the accumulation over time steps is",true],[". This is measured on the wavefield rather than the recorded shot, which is why the accumulated figures here are stricter than the shot-record numbers elsewhere.",false]]));

c.push(h2("Scoring the reflections, not the direct arrival"));
c.push(p("A single global number, in decibels or otherwise, is dominated by the strong direct arrival that runs along the top of the record. That arrival travels through the near-surface water layer, never enters the geology, and carries none of the image, so it is the easy part for any format to reproduce. To score the schemes on the part that matters, the direct arrival is removed."));
c.push(pm([["The method, proposed in the review meeting, is a water-model subtraction. ",true],["A second model is run that is water everywhere, at the surface velocity, with the source, receivers, geometry and time-stepping kept exactly the same and only the velocity changed. With no contrasts it produces the direct arrival and nothing else, computed accurately. Subtracting it from the real record removes the direct arrival and leaves the reflected energy on its own. The MX run is put through the water model too, so what it removes is the direct arrival MX itself produced.",false]]));
c.push(img("H2_water_gather.png",470,256));
c.push(cap("Figure 4b. The full record, the water model (direct arrival only), and their difference (the reflections). The strong arrival is gone from the difference, leaving the weak reflected energy."));
c.push(pm([["On this model the reflections carry about 45 percent of the record's energy and the direct arrival the other 55 percent, so the global number is roughly half made of the part we do not care about. Scored on the reflections alone, every scheme sits lower. ",false],["At 12 mantissa bits the global SNR is 49 dB, but the reflection-only SNR is 41 dB",true],[", about 8 dB stricter, and the same gap holds across bit widths.",false]]));
c.push(img("H1_water_snr.png",430,273));
c.push(cap("Figure 4c. Global SNR against reflection-only SNR, by bit width. The reflection figure is uniformly lower, since it drops the easy direct arrival."));
c.push(pm([["The ranking of the schemes does not change, but the reflection SNR is the honest figure to quote, ",false],["because it corresponds to what the imaging is actually built from",true],[". It is the metric the report will lead on going forward.",false]]));
c.push(img("D2_speedup_by_kernel.png",460,250));
c.push(cap("Figure 4. Predicted speedup of FP16 and MX at matched accuracy, across four kernel types."));
c.push(pm([["MX is ahead of FP16 on all four kernels. One result runs against expectation: ",false],["the elastic TTI kernel gains the least",true],[" at 1.53x, because it carries 22 model arrays of anisotropy parameters, 38 percent of its traffic, which are not compressed. The corollary is that ",false],["compressing the model arrays would pay off most on the anisotropic kernels",true],[".",false]]));

c.push(h2("The block-size anomaly is a shape effect"));
c.push(p("A block of 32 values was found to be more accurate than smaller blocks, which is the wrong way round if size were all that mattered. The explanation is shape. The blocks used everywhere else are one-dimensional strips: 32 values taken consecutively along the fastest storage axis, which runs down the field and can cross very different parts of the wavefront, so the shared exponent has to span a wide range of magnitudes. A block that is square in space gathers its values from a small compact patch, where the amplitudes are more alike, so the shared exponent fits them better."));
c.push(pm([["This was tested by holding the block size and bit cost fixed and changing only the shape. ",true],["Every block still holds about 32 values at the same bits per value; only the arrangement changes, from a 32 by 1 strip through to a 6 by 6 square.",false]]));
c.push(img("I1_block_shape.png",430,273));
c.push(cap("Figure 4d. Left, accuracy against block shape at fixed size and cost. Right, the pack and unpack time for each shape. Aspect ratio 1 is a square block, higher is more strip-like."));
c.push(pm([["The accuracy trend is clear and settles the question. ",true],["The memory strip sits at 23.9 dB, the square block at 28.5 dB, a gain of 4.6 dB at the same cost. Accuracy climbs steadily as the block becomes squarer, so ",false],["the anomaly is a shape effect, not a size effect",true],[": a compact block shares its exponent over more similar amplitudes. There is a second, smaller signal in the two 2-to-1 blocks, which differ by about 2 dB depending on which way they are oriented, a hint that the wavefronts have a preferred direction.",false]]));
c.push(h2("Does the shape cost speed?"));
c.push(pm([["The square block needs a two-dimensional tiling to pack and unpack, where the strip is a flat reshape, so the natural worry is that it is slower. Timing the pack and unpack directly, ",false],["the square block costs about 1.1 to 1.2 times the strip, a difference of a fraction of a millisecond per call",true],[". Two things make that negligible. The storage cost is identical, since every shape holds the same number of values at the same bits, so the memory traffic and therefore the modelled speed-up are the same for all shapes; shape changes accuracy, not bandwidth. And the solver is bandwidth-bound, so a small addition to the pack and unpack arithmetic is hidden behind the memory traffic it saves. These are numpy timings on a CPU and are indicative rather than final, but the margin is wide enough that the extra tiling cost is not a reason to avoid square blocks.",false]]));
c.push(pm([["The practical lesson, then. ",true],["Blocks should follow the geometry of the field rather than the layout of memory: a square block buys about 4.6 dB at the same storage cost and the same modelled speed, for a pack and unpack overhead too small to matter in a bandwidth-bound solver. The result points directly at cube-shaped blocks in three dimensions, and squaring the block is a low-risk change worth making once the small timing cost is confirmed on real hardware.",false]]));

c.push(pageBreak());
// Elastic
c.push(h1("Elastic kernel comparison"));
c.push(p("The acoustic comparison is not the setting Fabien-Ouellet's scaling was designed for. His work is elastic, where the wavefield carries several quantities whose magnitudes differ, which is the condition under which scaling is meant to help. The comparison was repeated on the elastic kernel, with all five wavefield components stored narrow at each step."));
c.push(img("E1_elastic_comparison.png",430,278));
c.push(cap("Figure 5. Elastic kernel, per-field storage accuracy (single-field level): MX against a per-field FP16 scaling, same model and metric. Right axis reads signal-to-noise ratio in decibels."));
c.push(pm([["The result holds, and holds at fewer bits than before. ",true],["At 14.25 bits MX reaches 8.75e-4, which is 61 dB, against 1.73e-3, or 55 dB, for the per-field scaled FP16 at 16 bits. Pushed harder, MX at 11 mantissa bits, only 13.25 bits per value, still reaches 1.34e-3, or 57 dB, above the FP16 point while spending nearly three bits per value less. MX's per-block exponent adapts to each quantity's amplitude on its own, doing the work the hand-chosen per-field factors are there to do, without the tuning.",false]]));
c.push(ti("Table 2. Measured accuracy and cost, elastic kernel. SNR in decibels is the primary accuracy column."));
c.push(table([["Scheme","Bits per value","SNR (dB)","Relative error"],["FP16, no scaling","16.00","54.9","1.80e-03"],["FP16, per field scaling","16.00","55.2","1.73e-03"],["MX, 10 mantissa bits","12.25","53.6","2.10e-03"],["MX, 11 mantissa bits","13.25","57.5","1.34e-03"],["MX, 12 mantissa bits","14.25","61.2","8.75e-04"],["MX, 14 mantissa bits","16.25","76.9","1.43e-04"]],[2900,1700,1300,1700]));
c.push(pm([["The crossover is worth naming. ",true],["MX matches the per-field FP16 point at about 10 to 11 mantissa bits and beats it clearly by 11, so 11 bits is the aggressive setting to reach for on this kernel: it clears the FP16 baseline and sits well above a 40 dB target while costing fewer bits.",false]]));
c.push(pm([["This figure and the efficiency figure are not in tension, though they can look it. ",true],["Figure 5 is a single-field measurement, the storage accuracy of the wavefield, and there MX is excellent. Figure 2 is a whole-solver measurement, the memory traffic of every array together, and there the speed-up is capped near 2x. The cap is set by the uncompressed model arrays, not by the wavefield's per-field accuracy; the two describe different levels of the system, and both are true at once.",false]]));
c.push(pm([["An honest limit on the baseline. ",true],["The per field scaling improves on naive FP16 by only about 4 percent here, because the stored velocity and stress values span only about a factor of five, which FP16 handles without much help. So the point compared against is a weak one in this setting: Fabien-Ouellet's scaling targets the dynamic range of the arithmetic, not the stored field values, and a storage-only comparison understates where it helps. The claim made here is therefore the narrower and fair one, that for wavefield storage MX's adaptive exponent matches or beats a per-field global scale at fewer bits.",false]]));

c.push(pageBreak());
// Visual evidence
c.push(h1("Visual evidence"));
c.push(h2("Receiver gather for each scheme"));
c.push(img("C3_trace_errors.png",470,259));
c.push(cap("Figure 6. A band of receiver traces in the wiggle display, one panel per scheme. The direct arrival is saturated so the weak reflections, which carry the image, are visible."));
c.push(p("The strong direct arrival that runs along the top is the least interesting part of the record, and on a natural amplitude scale it dominates the display and hides everything else. Here it is deliberately driven off scale and clipped flat, so the much weaker reflected and refracted energy underneath shows through. It is on those reflections that the schemes should be judged, since they are what an image is built from."));
c.push(p("Read that way, full precision, FP16 with scaling, and MX all reproduce the reflection pattern, but MX stays closest to the reference while FP16 and bfloat16 add more spurious wiggle to the weak arrivals, bfloat16 the most. A qualification belongs with this figure: the single relative error number quoted elsewhere is dominated by the strong direct arrival, so it mainly reports how well the strong, easy part of the record is preserved, not the weak reflections seen here. A measure aimed at the reflections would tell a sharper story, and choosing one is noted as future work."));
c.push(h2("Wavefield comparison"));
c.push(img("B3_wavefield_comparison.png",300,500));
c.push(cap("Figure 7. Reference wavefield, and the MX wavefield with its difference at 14 and at 10 mantissa bits. Each difference is drawn on its own much finer scale, stated in its colour bar, since on the wavefield scale it would be invisible."));
c.push(pm([["The wavefield and difference are on different scales, deliberately. ",true],["On the wavefield's own scale the MX field is indistinguishable from the reference and the difference is blank, which tells us only that the error is small. Drawing each difference on its own scale, tens to a hundred times finer, shows what the residual actually looks like.",false]]));
c.push(pm([["At 14 bits the residual is faint and lacks strong wavefront-following structure; at 10 bits it is several times larger and begins to trace the wavefronts, the sign of coherent distortion rather than harmless grain. ",false],["Showing the two bit widths together lets the residual be watched growing and organising itself as the format is pushed",true],[". Fabien-Ouellet (2020) reported FP16 error to be incoherent with the seismic signal in his elastic case; that this is only partly true for MX as it is pushed is a reason to prefer the higher bit widths, where the residual stays disorganised.",false]]));

c.push(pageBreak());
c.push(h2("Receiver gather"));
c.push(img("B4_trace_comparison.png",360,315));
c.push(cap("Figure 8. A band of receiver traces, full precision against MX at 14 mantissa bits, with the direct arrival saturated so the reflections are visible."));
c.push(p("The same saturated display as Figure 6, now full precision against MX alone. With the direct arrival clipped flat, the weak reflections stand out, and the two panels reproduce them in the same places with the same character. At this bit width MX preserves not just the strong arrival but the weak reflected energy the image depends on."));

// Baseline
c.push(h1("Baseline validation"));
c.push(p("These figures produce no results of their own; they establish that the baseline is sound and belong in the methods section."));
c.push(h2("Velocity model and acquisition"));
c.push(img("A1_velocity_model.png",470,180));
c.push(cap("Figure 9. Marmousi with sources (stars) and receivers (triangles)."));
c.push(h2("Shot record"));
c.push(img("A2_shot_record.png",295,253));
c.push(cap("Figure 10. The data recorded at the receivers for the central shot."));
c.push(pageBreak());
c.push(h2("Wavefield snapshot"));
c.push(img("A3_wavefield_snapshot.png",470,180));
c.push(cap("Figure 11. The wave propagating through the model, scattered by the layers and faults."));
c.push(h2("FP32 against FP64"));
c.push(img("A4_fp32_vs_fp64.png",450,253));
c.push(cap("Figure 12. The two traces coincide; their difference is 9.0e-4 over the record."));
c.push(p("This is the origin of the reference bar used throughout. FP32 already departs from FP64 by 9.0e-4, so an MX setting reaching this level is as accurate as the precision in production use."));

c.push(pageBreak());
// Reproduction
c.push(h1("Full reproduction of Fabien-Ouellet, arithmetic in FP16"));
c.push(p("Everything above narrows storage while computing in FP32. Fabien-Ouellet instead runs the arithmetic itself in FP16 and scales the equation to keep it in range. Reproducing him requires a solver that genuinely computes in 16 bits, including real overflow and underflow."));
c.push(pm([["Devito cannot do this. ",true],["Its code generator rejects a float16 grid, with the error that float16 cannot be converted to a ctypes type. So the reproduction is a small self-contained second order acoustic stepper written directly in numpy, run on the same cropped Marmousi velocities. numpy FP16 arithmetic overflows to infinity at 65504 and loses precision below its smallest normal near 6.1e-5, so the format's real behaviour is present rather than emulated. This stepper is independent of Devito and second order, so its numbers are not identical to the Devito runs; it establishes the precision behaviour, not a matched waveform.",false]]));
c.push(img("E2_fabien_reproduction.png",470,287));
c.push(cap("Figure 13. FP16 arithmetic (red) against MX storage (blue) in the same numpy acoustic stepper."));
c.push(ti("Table 3. Reproduction results. Arithmetic precision and storage cost listed separately."));
c.push(table([["Scheme","Arithmetic","Storage bits","Relative error"],["FP16, no scaling","FP16","16","1.09e-02"],["FP16 + scaling (Fabien-Ouellet)","FP16","16","9.69e-03"],["MX, 10 mantissa bits","FP32","12.25","1.28e-02"],["MX, 12 mantissa bits","FP32","14.25","2.74e-03"]],[3000,1500,1500,1700]));
c.push(pm([["The reproduction is correct. ",true],["FP16 arithmetic runs, and Fabien-Ouellet's scaling improves it, from 1.09e-2 to 9.69e-3. The improvement is modest because this is a single variable acoustic problem; as with the elastic comparison, scaling matters more when several quantities of different magnitude are present.",false]]));
c.push(pm([["The comparison is not like for like, and that is the finding. ",true],["MX at 14.25 bits reaches 2.74e-3, better than FP16 with scaling. But MX computes in FP32 and narrows only storage, while FP16 computes in 16 bits. ",false],["The two are different kinds of object",true],[". FP16 is a compute precision: it cuts arithmetic cost and memory together, limited to about 1e-2 accuracy and needing a scale factor. MX is a storage precision: the arithmetic stays exact, so accuracy per stored bit is higher, but it cuts memory only, not compute.",false]]));

c.push(pageBreak());
// Hybrid
c.push(h1("FP16 and MX combined"));
c.push(p("If FP16 narrows the arithmetic and MX narrows the storage, the obvious question is whether they combine: run the arithmetic in FP16, to save compute, and store the wavefield in MX below 16 bits, to save more memory than FP16 alone. The risk is that two lossy layers in one feedback loop compound, since FP16 rounding and MX rounding both enter every step and are fed back."));
c.push(p("This was tested in the same numpy stepper: FP16 arithmetic with scaling, and MX storage on top, swept across storage bit widths, against pure MX and pure FP16 as controls."));
c.push(img("E3_hybrid.png",460,298));
c.push(cap("Figure 14. Pure MX (blue), the hybrid of FP16 arithmetic plus MX storage (purple), and pure FP16 (dashed red). Lower is better."));
c.push(ti("Table 4. Hybrid against its two components, by storage bit width."));
c.push(table([["Storage bits","Pure MX (FP32)","Hybrid (FP16 + MX)","Pure FP16"],["8.25","2.53e-01","2.38e-01","9.69e-03"],["10.25","5.14e-02","4.98e-02","9.69e-03"],["12.25","1.28e-02","1.55e-02","9.69e-03"],["14.25","2.74e-03","8.14e-03","9.69e-03"]],[2000,2100,2300,1600]));
c.push(pm([["The two do not compound. ",true],["The hybrid error tracks whichever layer is worse, not the sum of the two. Where MX storage is coarse it sits on the pure MX line; where MX storage is fine it flattens onto the pure FP16 floor near 9e-3, set by the 16 bit arithmetic. At its best point the hybrid error is 0.84 times the larger of its two component errors, so combining them adds nothing and destabilises nothing.",false]]));
c.push(pm([["The practical consequence. ",true],["FP16 arithmetic and MX storage can be used together. Running the arithmetic in FP16 gives the compute saving, and storing the wavefield in MX at about 12 mantissa bits gives a memory footprint below FP16's 16 bits, at the accuracy FP16 alone delivers. There is no benefit to storing MX more finely than that, since below the FP16 floor the arithmetic dominates. This is a concrete design point: FP16 for the compute path, MX at roughly 12 to 14 bits for the stored wavefield.",false]]));
c.push(ti("Table 5. The three approaches side by side, at their best comparable setting."));
c.push(table([["Approach","What it narrows","Saves","Relative error","Storage bits"],["Pure FP16 + scaling","arithmetic","compute and memory","9.69e-03","16.00"],["Pure MX","storage","memory only","2.74e-03","14.25"],["Hybrid, FP16 + MX","both","compute and memory","8.14e-03","14.25"]],[2600,1900,2200,1500,1300]));
c.push(pm([["Read this way: ",false],["pure MX is the most accurate",true],[", because its arithmetic is exact, but it saves memory only. ",false],["Pure FP16 saves compute as well",true],[", at lower accuracy and needing a scale factor. ",false],["The hybrid keeps FP16's compute saving and adds MX's memory saving below 16 bits",true],[", at the accuracy the FP16 arithmetic allows. Which one is right depends on whether the bottleneck is memory alone or compute as well; none dominates the other on every axis.",false]]));

c.push(pageBreak());
// int16 analysis
c.push(h1("Why int16 underperforms, and how it improves"));
c.push(p("int16 with scaling was expected to beat FP16, and on the face of it should: FP16 spends five of its sixteen bits on an exponent and keeps only ten of mantissa, while int16 is fixed point and keeps fifteen. For values near the field maximum that holds, and int16 resolves them more finely than FP16. It nonetheless comes out worse overall, at 8.09e-3 against 6.13e-3, and the reason is the wavefield's dynamic range."));
c.push(pm([["A fixed point format has one step size for the whole field, set by the largest value. ",true],["The wavefield spans a wide range of magnitudes: the strong direct arrival is more than a thousand times the median amplitude. Measured against a single int16 step, a value near the peak keeps about twelve bits, a mid amplitude value about nine, and a median value only about three, while the weakest are lost entirely. FP16, being floating point, keeps about ten bits of relative precision at every magnitude. So across the great majority of the field, which is far weaker than the direct arrival, FP16 resolves values that int16 cannot, and it wins the average even though it loses at the peak.",false]]));
c.push(pm([["The fix follows directly, and it is the project's own scheme. ",true],["The problem is a single scale spanning the whole range, so the remedy is more scales: one per block, each set by its own local maximum, so weak regions get a fine step and strong regions a coarse one and every value keeps its relative precision. Giving int16 per block scaling is not a patch on a different method; it is microscaling. The global int16 scheme is simply the limit of MX with the block grown to the whole grid.",false]]));
c.push(img("F1_int16_improvement.png",430,281));
c.push(cap("Figure 15. int16 error as the number of independent scales increases. One scale is the usual global int16; more scales is per block scaling, which is microscaling."));
c.push(pm([["The figure makes it concrete. ",true],["With one global scale int16 sits at 8.09e-3, above FP16. Adding per block scaling drops it below FP16 immediately, and it keeps falling as the blocks shrink, passing into the microscaling range. At block 32 the per block int16 reaches 5.7e-4, though at a higher bit cost than MX at twelve mantissa bits, since it carries fifteen; the two are the same block floating point family separated only by mantissa width. The lesson is that int16 does not underperform because fixed point is inferior, but because one scale cannot span the dynamic range, and removing that single assumption turns it into the scheme this project studies.",false]]));
c.push(h2("Established"));
c.push(bulletm([["MX beats FP16 as a storage format, on both kernels. ",true],["At 14.25 bits MX is cheaper and more accurate than FP16 at 16 bits, on the acoustic kernel (3.37e-3 against 6.13e-3) and the elastic kernel (8.75e-4 against 1.73e-3), the latter on the physics FP16 scaling was designed for."]]));
c.push(bulletm([["FP16 and MX are complementary and combine cleanly. ",true],["Fabien-Ouellet's FP16 arithmetic method was reproduced and verified. FP16 is a compute precision and MX a storage precision; combining them does not compound the error, so FP16 arithmetic with sub-16-bit MX storage is a usable design point."]]));
c.push(bulletm([["The comparison is defensible. ",true],["Every scheme was reimplemented in the same harness and run on the same model with the same metric, rather than quoted from papers using different models and measures."]]));
c.push(bulletm([["The governing mechanism is identified. ",true],["The constraint is error accumulation over the time stepping, not single frame representation error."]]));
c.push(h2("Limitations, in order of importance"));
c.push(bulletm([["The speedup is capped at about 2x. ",true],["Only the wavefield is compressed and the model arrays remain FP32, which bounds the gain regardless of format. Compressing the model arrays would raise the ceiling to roughly 5x, at low risk since model parameters are static and do not accumulate error."]]));
c.push(bulletm([["The quantiser is a simplification. ",true],["It uses a shared exponent with a fixed point code. The OCP MXFP standard gives each element its own small exponent too. Guidance on safe settings cannot be published against a non standard format, so this must be replaced."]]));
c.push(bulletm([["Efficiency is modelled, not measured. ",true],["A wall clock measurement is not meaningful under numpy emulation. On suitable hardware a micro kernel should verify the bandwidth model."]]));
c.push(bulletm([["Two results remain unexplained. ",true],["Stochastic rounding does not outperform round to nearest, the opposite of Croci and Giles for the parabolic heat equation, and quantising less frequently has not been swept. The block-size anomaly, by contrast, is now explained: it is a shape effect, with square blocks more accurate than memory strips at the same cost."]]));
c.push(bulletm([["Scope. ",true],["The sweeps are 2D and single shot on cropped models, and the FP16 arithmetic reproduction uses an independent second order numpy stepper. Full Marmousi runs take about four times as many steps, so accumulation is worse and the safe bit width higher; final settings must be confirmed against it."]]));
c.push(h2("Next steps"));
c.push(bulletm([["Compress the model arrays. ",true],["Small effort, low risk, lifts the speedup ceiling from 2x towards 5x, and untested in the literature. This is the immediate priority."]]));
c.push(bulletm([["Move to the exact OCP MXFP format. ",true],["So that the safe bit width figures correspond to a real, publishable format."]]));
c.push(bulletm([["Follow up the block-shape result. ",true],["Square blocks beat memory strips at the same cost, so the immediate follow-ups are cube-shaped blocks in three dimensions and the small orientation effect seen between the two rectangular blocks. Separately, sweep the quantisation interval to see how far suppressing accumulation lowers the safe bit width."]]));

c.push(h2("Running on a GPU"));
c.push(p("The natural home for this work is the GPU, and taking it there is the step that would turn the predicted saving into a measured one. It is worth being precise about what changes and what does not, because the distinction is what makes the present results trustworthy in the first place."));
c.push(pm([["What this study does. ",true],["The MX format is emulated in numpy: at each step the wavefield is packed to MX and unpacked back, so the number that comes out is the accuracy the format would give. The arithmetic itself stays FP32, and no bytes are actually saved; the memory saving is computed from a bandwidth model rather than observed. So the accuracy is real and exact, while the efficiency is a prediction.",false]]));
c.push(pm([["What a GPU implementation changes. ",true],["The pack and unpack would move into a GPU kernel and fuse with the solver, so a value is compressed to MX as it is written to memory and expanded as it is read back. At that point the format genuinely reduces the bytes moved through device memory, the FP16 arithmetic baseline could run in place rather than in a separate numpy stepper, and the speed-up would be timed on the wall clock instead of modelled. The pack and unpack overhead measured here on a CPU would also be re-measured, where it is likely to be hidden, since on a GPU those operations are highly parallel and overlap with the memory traffic they serve.",false]]));
c.push(pm([["What does not change. ",true],["The accuracy results carry over unaltered. How much precision a value loses when stored in a given number of bits is independent of the hardware, so every SNR figure, the reflection-only scores, the bit-width sweeps and the block-shape result would be identical on a GPU. So would the mechanism: error accumulation as the binding constraint, the int16 diagnostic, and why MX beats FP16. The GPU changes how the efficiency is obtained, not the accuracy findings the work rests on.",false]]));
c.push(pm([["Why the prediction should hold. ",true],["The efficiency claim rests on one property, that the solver is memory-bandwidth-bound: its speed is set by how many bytes move, not how many operations run. Under that condition the speed-up is essentially the byte-compression ratio, which is exactly what the bandwidth model computes. A GPU does not weaken this, it strengthens it: GPUs are more bandwidth- and capacity-bound than CPUs, and 3D models often do not fit in device memory at all, so a storage format that moves fewer bytes helps more there, not less. The one thing the model cannot see is whether the pack and unpack, once in a real kernel, are cheap enough not to erode the saving; the CPU timing here suggests they are small, and confirming it on hardware is precisely the measurement a GPU run would add.",false]]));

const doc=new Document({
  styles:{default:{document:{run:{font:FONT,size:21}}}},
  numbering:{config:[{reference:"bul",levels:[{level:0,format:LevelFormat.BULLET,text:"\u25AA",alignment:AlignmentType.LEFT,style:{run:{color:NAVY},paragraph:{indent:{left:400,hanging:220}}}}]}]},
  sections:[{properties:{page:{margin:{top:1000,right:1080,bottom:1000,left:1080}}},children:c}],
});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("/tmp/report_final.docx",b);console.log("written",b.length);});
