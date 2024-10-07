using MPI

mpiexec() do cmd
    
    run(`$cmd -n 2 $(Base.julia_cmd()) --threads=1 --project=@. -e
    'include("6_Applications/6_3_IsentropicVortexAdvection/6_3_1_MPI/elixir_euler_vortex_PERK4.jl")'`)
end
