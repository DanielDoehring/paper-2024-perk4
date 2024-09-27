# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Custom implementation for PERK integrator
function (amr_callback::AMRCallback)(integrator::Union{PERK_Multi_Integrator,
                                                       PERK3_Multi_Integrator,
                                                       PERK4_Multi_Integrator};
                                     kwargs...)
    u_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "AMR" begin
        has_changed = amr_callback(u_ode, semi,
                                   integrator.t, integrator.iter; kwargs...)

        if has_changed
            resize!(integrator, length(u_ode))
            u_modified!(integrator, true)

            ### PERK additions ###
            # TODO: Need to make this much less allocating!
            @trixi_timeit timer() "PERK stage identifiers update" begin
                mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

                integrator.n_levels = get_n_levels(mesh, integrator.alg)
                n_dims = ndims(mesh) # Spatial dimension

                # Re-initialize storage for level-wise information
                if integrator.n_levels != length(integrator.level_info_elements_acc)
                    integrator.level_info_elements = [Vector{Int64}()
                                                        for _ in 1:integrator.n_levels]
                    integrator.level_info_elements_acc = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]

                    integrator.level_info_interfaces_acc = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]
                    integrator.level_info_mpi_interfaces_acc = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]

                    integrator.level_info_boundaries_acc = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]
                    # For efficient treatment of boundaries we need additional datastructures
                    integrator.level_info_boundaries_orientation_acc = [[Vector{Int64}()
                                                                            for _ in 1:(2 * n_dims)]
                                                                            # Need here n_levels, otherwise this is not Vector{Vector{Int64}} but Vector{Vector{Vector{Int64}}
                                                                        for _ in 1:integrator.n_levels]
                    integrator.level_info_mortars_acc = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]
                    integrator.level_info_mpi_mortars_acc = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]  
                                                          
                    integrator.level_u_indices_elements = [Vector{Int64}()
                                                            for _ in 1:integrator.n_levels]
                else # Just empty datastructures
                    for level in 1:integrator.n_levels
                        empty!(integrator.level_info_elements[level])
                        empty!(integrator.level_info_elements_acc[level])

                        empty!(integrator.level_info_interfaces_acc[level])
                        empty!(integrator.level_info_mpi_interfaces_acc[level])

                        empty!(integrator.level_info_boundaries_acc[level])
                        for dim in 1:(2 * n_dims)
                            empty!(integrator.level_info_boundaries_orientation_acc[level][dim])
                        end

                        empty!(integrator.level_info_mortars_acc[level])
                        empty!(integrator.level_info_mpi_mortars_acc[level])

                        empty!(integrator.level_u_indices_elements[level])
                    end
                    empty!(integrator.level_info_elements[integrator.n_levels])
                    empty!(integrator.level_u_indices_elements[integrator.n_levels])
                end

                
                partitioning_variables!(integrator.level_info_elements, 
                                        integrator.level_info_elements_acc, 
                                        integrator.level_info_interfaces_acc, 
                                        integrator.level_info_boundaries_acc, 
                                        integrator.level_info_boundaries_orientation_acc,
                                        integrator.level_info_mortars_acc,
                                        integrator.n_levels, n_dims, mesh, dg, cache, integrator.alg)
                
                
                #=
                partitioning_variables!(integrator.level_info_elements, 
                                        integrator.level_info_elements_acc, 
                                        integrator.level_info_interfaces_acc,
                                        integrator.level_info_mpi_interfaces_acc,
                                        integrator.level_info_boundaries_acc, 
                                        integrator.level_info_boundaries_orientation_acc,
                                        integrator.level_info_mortars_acc,
                                        integrator.level_info_mpi_mortars_acc,
                                        integrator.n_levels, n_dims, mesh, dg, cache, integrator.alg)
                =#

                #=
                # NOTE: Optional RHS computation (for PERK4 paper)
                Stages = [19, 11, 7, 5] # Isentropic Vortex with 4Lvl AMR
                for level = 1:length(integrator.level_info_elements)
                    integrator.AddRHSCalls += amr_callback.interval * Stages[level] * 
                                                length(integrator.level_info_elements[level])
                end
                =#

                partitioning_u!(integrator.level_u_indices_elements, integrator.n_levels, n_dims, integrator.level_info_elements, 
                                u_ode, mesh, equations, dg, cache)
            end # "PERK stage identifiers update" timing
        end # if has changed
    end # "AMR" timing

    return has_changed
end
end # @muladd
