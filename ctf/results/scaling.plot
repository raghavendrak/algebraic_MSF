set ylabel "Time (seconds)"
set xlabel "Cores of Stampede2"
set x2label "Scale of R-MAT graph (log #vertices, 8 edges/vertex)"
set title "Weak Scaling of Graph Connectivity using Cyclops"
set logscale x 2
#set logscale x2 2
#set logscale y 2
set key top left
set x2range [17:21];
set xrange [64:1024];
set yrange [.5:12];


#set format x "%1.1E"


set pointsize 0.8

set format x "%g"
#set format y "%1.3f"
#set size 0.85, 0.8
#model line style
set style line 25 lt 1 lc rgb "#015223" lw 3 pt 10 ps 2.0
set style line 26 lt 1 lc rgb "#415223" lw 3 pt 12 ps 2.0
#grid line style
#set style line 99 lt 2 lc rgb "gray" lw 1 pt 1

load "cfg.plot"


set x2tics 17,1
set xtics 2
set mxtics 2
set ytics 2
set mytics 2
set grid xtics mxtics ytics mytics ls 99

set output "scaling.eps"

plot "scaling.dat" using (64*$1):5 ls 1 with linespoints title 'Shiloach-Vishkin Algorithm', \
     "scaling.dat" using (64*$1):6 ls 13 with linespoints title 'Recursive Projection Algorithm'

