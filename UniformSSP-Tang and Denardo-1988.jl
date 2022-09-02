## Problem: Job sequencing and tool switching problem (SSP) - Tang and Denardo (1988) formulation
## Solver: Gurobi
## Language: Julia (JuMP)
## Written by: @setyotw
## Date: Sept 2, 2022

#%% import packages
using Pkg, JuMP, Gurobi, DataStructures
Pkg.status()
import FromFile: @from

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS
function UniformSSP_TangDenardo_Formulation(instanceSSP, magazineCap, MILP_Limit)
    # 1 | initialize sets and notations
    # number of available jobs (horizontal)
    n = length(instanceSSP[1,:])
    # number of available tools (vertical)
    m = length(instanceSSP[:,1])

    J = [i for i in range(1, n)] ## list of jobs
    K = [i for i in range(1, n)] ## list of positions
    T = [i for i in range(1, m)] ## list of tools
    arcJK = [(j,k) for j in J for k in K]
    arcKT = [(k,t) for k in K for t in T]
    Tj = Dict((j) => [] for j in J)
    for job in J
        for tools in T
            if instanceSSP[tools,job] == 1
                append!(Tj[job], tools)
            end
        end
    end

    Jt = Dict((t) => [] for t in T)
    for tools in T
        for job in J
            if instanceSSP[tools,job] == 1
                append!(Jt[tools], job)
            end
        end
    end

    # 2 | initialize parameters
    C = Int(magazineCap)
    
    # 3 | initialize the model
    model = Model(Gurobi.Optimizer)

    # 4 | initialize decision variables
    @variable(model, U[arcJK], Bin) # U[jk] = Equal to 1 if job j processed in position k
    @variable(model, V[arcKT], Bin) # V[kt] = Equal to 1 if tool t presents while performing a job in position k
    @variable(model, W[arcKT], Bin) # W[kt] = tool switch, equal to 1 if tool t is in magazine while performing a job
    
    # 5 | define objective function
    @objective(model, Min, 
        sum(V[(2,t)] for t in T) + sum(W[(k,t)] for k in K for t in T if k!=1))

    # 6 | define constraints
    # (1) sum-j-in-range-J Ujk = 1, for k in range K ##
    for k in K
        @constraint(model, sum(U[(j,k)] for j in J) == 1)
    end
    
    # (2) sum-k-in-range-K Ujk = 1, for j in range J ##
    for j in J
        @constraint(model, sum(U[(j,k)] for k in K) == 1)
    end

    # (3) sum-j-in-range-Jt Ujk <= Vkt, for k in range K and t in range T ##
    for k in K
        for t in T
            @constraint(model, sum(U[(j,k)] for j in Jt[t]) <= V[(k,t)])
        end
    end

    # (4) sum-t-in-range-T Vkt <= C, for k in range K ##
    for k in K
        @constraint(model, sum(V[(k,t)] for t in T) <= C)
    end

    # (5) Vkt - V(k-1,t) <= Wkt, for k in range K,(k=/1) and t in range T ##
    for k in K
        for t in T
            if k!=1
                @constraint(model, V[(k,t)] - V[(k-1,t)] <= W[(k,t)])
            end
        end
    end

    # 7 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    JuMP.set_time_limit_sec(model, MILP_Limit)
    JuMP.optimize!(model)

    # 8 | extract the results    
    completeResults = solution_summary(model)
    solutionObjective = objective_value(model)
    solutionGap = relative_gap(model)
    runtimeCount = solve_time(model)
    all_var_list = all_variables(model)
    all_var_value = value.(all_variables(model))
    U_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'U']
    V_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'V']
    W_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'W']
    
    return solutionObjective, solutionGap, U_active, V_active, W_active, runtimeCount, completeResults
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple uniform SSP case with 5 different jobs, 6 different tools, and 3 capacity of magazine (at max, only 3 different tools could be installed at the same time)
instanceSSP = Array{Int}([
        1 1 0 0 1;
        1 0 0 1 0;
        0 1 1 1 0;
        1 0 1 0 1;
        0 0 1 1 0;
        0 0 0 0 1])

magazineCap = Int(3)

#%% termination time for the solver (Gurobi)
MILP_Limit = Int(3600)

#%% implement the mathematical formulation
# solutionObjective --> best objective value found by the solver
# solutionGap --> solution gap, (UB-LB)/UB
# U_active, V_active, W_active --> return the active variables
# runtimeCount --> return the runtime in seconds
# completeResults --> return the complete results storage
solutionObjective, solutionGap, U_active, V_active, W_active, runtimeCount, completeResults = UniformSSP_TangDenardo_Formulation(instanceSSP, magazineCap, MILP_Limit)