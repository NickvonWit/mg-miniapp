#!/bin/bash
#SBATCH --job-name=mg3d_bench
#SBATCH --output=mg3d_bench_%j.out
#SBATCH --error=mg3d_bench_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodelist=piora2

SERIAL_EXE=./build/mg3d_serial
KOKKOS_EXE=./build/mg3d_kokkos_ai

SIZES=("33 33 33" "65 65 65" "129 129 129" "193 193 193" "257 257 257")
CORES=(1 2 4 8 16 32 64 128)
REPEATS=3

OUTFILE="benchmark_results_${SLURM_JOB_ID}.csv"

echo "type,nx,ny,nz,threads,run,iters,rel_residual,rel_L2_error,wall_time_s" > "$OUTFILE"

parse_output() {
    local out="$1"
    local iters rel_res rel_l2 wtime
    iters=$(echo "$out" | grep "PCG iters" | sed 's/.*PCG iters: \([0-9]*\).*/\1/')
    rel_res=$(echo "$out" | grep "PCG iters" | sed 's/.*final rel. residual: \([0-9.e+-]*\).*/\1/')
    rel_l2=$(echo "$out" | grep "PCG iters" | sed 's/.*rel. L2 error: \([0-9.e+-]*\).*/\1/')
    wtime=$(echo "$out" | grep "Wall time" | sed 's/.*Wall time: \([0-9.]*\).*/\1/')
    echo "${iters},${rel_res},${rel_l2},${wtime}"
}

echo "=============================="
echo " MG3D Benchmark"
echo " Sizes: ${#SIZES[@]}, Cores: ${CORES[*]}, Repeats: $REPEATS"
echo "=============================="

# --- Sequential runs ---
echo ""
echo ">>> Sequential executable"
for size in "${SIZES[@]}"; do
    read -r nx ny nz <<< "$size"
    for r in $(seq 1 $REPEATS); do
        echo "  Serial: nx=$nx ny=$ny nz=$nz  [run $r/$REPEATS]"
        output=$($SERIAL_EXE -nx "$nx" -ny "$ny" -nz "$nz" 2>&1)
        parsed=$(parse_output "$output")
        echo "serial,$nx,$ny,$nz,1,$r,$parsed" >> "$OUTFILE"
    done
done

# --- Kokkos OpenMP runs ---
echo ""
echo ">>> Kokkos OpenMP executable"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

for size in "${SIZES[@]}"; do
    read -r nx ny nz <<< "$size"
    for ncores in "${CORES[@]}"; do
        export OMP_NUM_THREADS=$ncores
        for r in $(seq 1 $REPEATS); do
            echo "  Kokkos: nx=$nx ny=$ny nz=$nz  threads=$ncores  [run $r/$REPEATS]"
            output=$($KOKKOS_EXE -nx "$nx" -ny "$ny" -nz "$nz" 2>&1)
            parsed=$(parse_output "$output")
            echo "kokkos,$nx,$ny,$nz,$ncores,$r,$parsed" >> "$OUTFILE"
        done
    done
done

echo ""
echo "=============================="
echo " Done. Results in: $OUTFILE"
echo "=============================="

