from typing import Union
from json import dumps
import argparse
import cloudpickle
import json
import os
import sys
import datetime as dt
import time
import pandas as pd
import numpy as np
import gurobipy as gp   #Import library as gp in order to know which functions are from gurobi
from gurobipy import GRB
import matplotlib.pyplot as plt
import httpx
from pydantic import parse_obj_as
from fastapi.encoders import jsonable_encoder
from agricore_sp_models.simulation_models import *
from agricore_sp_models.agricore_sp_models import *
from agricore_sp_models.logging import configure_orm_logger
from agricore_sp_models.logging import configure_orm_logger
from typing import List
from collections import defaultdict
from loguru import logger
from contextlib import nullcontext
import math
import dask
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask import config as cfg
from bisect import bisect_left
import cProfile


from settings import settings

def test_production_data(populationId:int, yearNumber:int, download:bool = True, save:bool = False) -> Union[DataToLPDTO, None]:
    try:
        dump_file:str = f"response_object_{yearNumber}.json"
        ok_status:bool = False
        response_object:Union[DataToLPDTO, None] = None
        if (download):
            url:str = f'{settings.ORM_ENDPOINT}/population/{populationId}/farms/get/simulationdata/longperiod'
            params = {'year': yearNumber}
            client = httpx.Client()
            headers = {"Accept-Encoding": "gzip"}
            response = client.get(url, headers = headers, params=params, timeout=None)
            if response.status_code == 200:
                response_json = response.json()
                response_object = parse_obj_as(DataToLPDTO, response_json)
                if (save):
                    with open(dump_file, "w") as f:
                        f.write(dumps(response_object.dict()))
                ok_status = True
        else:
            with open(dump_file, "r") as f:
                response_json = json.load(f)
                response_object = parse_obj_as(DataToLPDTO, response_json)
                ok_status = True            
        return response_object, yearNumber
        
    except Exception as e:
        print(e)
        return None
    
def chunks(lst, n):
    """
        Yield successive n-sized chunks from lst.

        Args:
            lst (list): The list to be divided into chunks.
            n (int): The size of each chunk.

        Yields:
            list: Successive n-sized chunks from the original list.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def list_check_input(input):
    """
        Ensure the input is a list. If the input is not a list, convert it to a list.

        Args:
            input: The input to be checked and converted.

        Returns:
            list: The input converted to a list if it wasn't already a list.
    """
    if isinstance(input, list):
        pass
    else:
        input = [input]
    return input

def LP_algorithm(inputs, sim_period, LP_second_phase = False, LMM_transactions = []):
    """
        Linear Programming Algorithm for farm management optimization.

        Args:
            inputs (dict): Dictionary containing the initial parameters for the optimization.
            sim_period (int): The simulation period.
            LP_second_phase (bool, optional): Flag to indicate if it's the second phase of LP. Default is False.
            LMM_transactions (list, optional): List of transactions for LMM. Default is an empty list.

        Returns:
            dict: A dictionary containing the results of the optimization, which may include various metrics or outputs based on the optimization.
    """
    from loguru import logger
    try:
        # Check if Total current assets > 0
        if inputs["Total_current_assets"] <= 0:
            agents_id = inputs["agents_id"]
            TCA = inputs["Total_current_assets"]

            if inputs["Total_current_assets"] == 0:
                logger.debug(f"Agent {agents_id}: Total current assets = {TCA} == 0")
            else:
                logger.debug(f"Agent {agents_id}: Total current assets = {TCA} < 0")

            return "error"


        # ---- Extract LP inputs
        T_CA_0 = list_check_input(inputs["Total_current_assets"])
        A_0 = list_check_input(inputs["A_0"])
        L_Loans_0 = list_check_input(inputs["Long_loans"])
        FNI_0 = list_check_input(inputs["Farm_net_income"])
        GFI_0 = list_check_input(inputs["Gross_farm_income"])

        agents_id = list_check_input(inputs["agents_id"])
        agents_regions = list_check_input(inputs["agents_region"])

        land_ha_0 = list_check_input(inputs["land_ha"])
        land_avg_prod = inputs["land_avg_prod"]
        land_avg_costs = (inputs["land_avg_costs"])
        average_ha_va = inputs["average_ha_va"]

        holder_age = inputs["holder_age"]
        holder_suc = inputs['holder_succssesor']
        holder_suc_age = inputs['holder_succssesor_age']

        year = inputs["year"]

        AversionRisk = inputs["AversionRisk"]

        rentArea = inputs["rentArea"]
        rentValue = inputs["rentValue"]


        nr_agents = len(T_CA_0)

        # ---- Simulation parameters
        sim_period = sim_period + 1             # Simulation period
        i = sim_period*2               
        shape = [sim_period, nr_agents]         # Shape of the final matrixes

        VAT = 5000          # Taxes
        NFM = inputs["NFM"] # Number of Members in the family 
        EFT = 5000          # External factors

        R = 0.005           #Interest
        MEPI = 5000         #Multiple Effects Public Income (MEPI) indicator - minimum annual living income established by the national government for year 
        ML = 15             #Average maturity of the loans

        # Production estimation function
        X = np.array([*range(0, i + 1)])
        C_ECON = 10                                             #Medium length of the economical cycle
        PYLD = 0.15 + 0.05 * np.sin(2 * np.pi * X / (C_ECON))   #Average ratio of value of production per value of land
        
        ## ---- Matrix to save results
        a_f = np.zeros(shape)          # Matrix to save final values of A for all agents
        lt_f = np.zeros(shape)         # Matrix to save final values of LT for all agents
        ca_f = np.zeros(shape)         # Matrix to save final values of CA for all agents
        fa_f = np.zeros(shape)         # Matrix to save final values of FA for all agents
        d_f = np.zeros(shape)          # Matrix to save final values of D for all agents
        lr_f = np.zeros(shape)         # Matrix to save final values of LR for all agents
        e_f = np.zeros(shape)          # Matrix to save final values of E for all agents
        sr_f = np.zeros(shape)         # Matrix to save final values of SR for all agents
        fni_f = np.zeros(shape)        # Matrix to save final values of FNI for all agents
        gfi_f = np.zeros(shape)        # Matrix to save final values of GFI for all agents

        a_ha_f = np.zeros(shape)       # Matrix to save final values of Land size (ha) for all agents
        a_va_f = np.zeros(shape)       # Matrix to save final values of Land value (€/ha) for all agents
        prod_ha_f = np.zeros(shape)    # Matrix to save final values of Land production (€/ha) for all agents
        u1_ha_f = np.zeros(shape)      # Matrix to save final values of land, in ha, for sell/buy

        u1_f = np.zeros(shape)  
        u2_f = np.zeros(shape)   
        np_f = np.zeros(shape)         # Matrix to save final values of NP for all agents
        year_f = np.zeros(shape)
        
        ##------------ MODEL Iniciatilization ------------##
    
        with gp.Env() as env, gp.Model("Financial Model", env=env) as model:
            u1 = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float("-inf"), name = "Decision variable 1 - Buy/sell land")
            u2 = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float("-inf"), name = "Decision variable 2 - Loans")
            ca = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float(0), name = "Current Assets")
            fa = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float("-inf"), name = "Fixed Assets")
            lt = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float(0), name = "Long term liabilities")
            d = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float(0), name = "Deposits")
            a = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float(0), name = "Owned Land")
            fni = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float("-inf"), name = "Farm Net Income")
            gfi = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float("-inf"), name = "Gross Farm Income")
            e = model.addVars(i + 1, nr_agents, vtype = GRB.CONTINUOUS, lb=float("-inf"), name = "Equity")

            lr = np.zeros((i + 1, nr_agents))   # Liquidity ratio
            sr = np.zeros((i + 1, nr_agents))   # Solvency ratio
            netp = np.zeros((i + 1, nr_agents)) # Net pofitability

            # Land value and production
            A_ha = np.zeros((i + 1, nr_agents))     # Land size (ha)
            A_va = np.zeros((i + 1, nr_agents))     # Land ha value (€/ha)
            prod_ha = np.zeros((i + 1, nr_agents))  # Land production (€/ha)
            u1_ha = np.zeros((i + 1, nr_agents))    # Sell/buy land (ha)

            # Initialise variables #
            for agent_j in range(nr_agents):
                u1[0, agent_j] = 0
                u2[0, agent_j] = 0

                ca[0, agent_j] = T_CA_0[agent_j]
                d[0, agent_j] = ca[0, agent_j]
                fa[0, agent_j] = 0 if np.isnan(A_0[agent_j]) else A_0[agent_j]
                lt[0, agent_j] = L_Loans_0[agent_j]
                a[0, agent_j] = 0 if np.isnan(A_0[agent_j] ) else A_0[agent_j] 
                fni[0, agent_j] = FNI_0[agent_j]
                gfi[0, agent_j] = GFI_0[agent_j]
                e[0, agent_j] = fa[0, agent_j] + ca[0, agent_j] - lt[0, agent_j]
                lr[0, agent_j] = ca[0, agent_j] / (1 / ML) * lt[0, agent_j]

                if lt[0, agent_j] != 0:
                    sr[0, agent_j] = (lt[0, agent_j]) / (fa[0, agent_j] + ca[0, agent_j] - lt[0, agent_j])
                else:
                    sr[0, agent_j] = 0

                A_ha[0, agent_j] = land_ha_0[agent_j]

                if A_ha[0, agent_j] != 0:
                    A_va[0, agent_j] = a[0, agent_j] / A_ha[0, agent_j]
                    prod_ha[0, agent_j] = gfi[0, agent_j] / A_ha[0, agent_j]
                else:
                    # The agent does not have land
                    A_va[0, agent_j] = average_ha_va 
                    prod_ha[0, agent_j] = 0

                if LP_second_phase == True:
                    u1[1, agent_j] = 0 # Assume there are no transactions
                    # Extract agent's transactions in the LMM (if any)
                    agent_transactions = LMM_transactions
                    # agent_transactions = [transaction for transaction in LMM_transactions if transaction['Buyer_FarmId'] == agents_id[agent_j] or transaction['Seller_FarmId'] == agents_id[agent_j]]
                    # we assume now that the transactions are only of the agent, to avoid passing all the transactions each time
                    # If the agent got any transactions update u1 and land
                    if len(LMM_transactions) > 0:
                        if agent_transactions[0]['Buyer_FarmId'] == agents_id[agent_j]:
                            buyer_role = True
                        else:
                            buyer_role = False
                        if buyer_role == True:
                            u1[1, agent_j] = - agent_transactions[0]["SalePrice"]
                            A_ha[1, agent_j] = land_ha_0[agent_j] - agent_transactions[0]["Land_exchange"] 
                        elif buyer_role == False:
                            u1[1, agent_j] = agent_transactions[0]["SalePrice"]
                            A_ha[1, agent_j] = land_ha_0[agent_j] + agent_transactions[0]["Land_exchange"]
            
            ## ------------ MODEL UPTADE ------------ ##
            model.update()
            model

            # ------------ Optimization process ------------ #
            # -- Simulate each year 
            for c_year in range(1, sim_period):
                y = [0 for _ in range(nr_agents)]
                y1 = [0 for _ in range(nr_agents)]

                for pred_year in range(c_year, c_year + sim_period + 1):
                    for agent_j in range(nr_agents):
                        #Check subsidies vigor
                        inputs["SBT"]
                        SBT_c, SBT_nc = subsidies_check(inputs["SBT"], pred_year)
             
                        # ---- Equation ---- #
                        a[pred_year, agent_j] = a[pred_year - 1, agent_j] + u1[pred_year, agent_j] + rentValue
                        fa[pred_year, agent_j] = a[pred_year, agent_j]                    # Total fixed assets

                        lt[pred_year, agent_j] = (1 - 1 / ML) * lt[pred_year - 1, agent_j] + u2[pred_year, agent_j]     # Long term liabilities

                        gfi[pred_year, agent_j] = PYLD[pred_year] * a[pred_year, agent_j]               # Gross Farm Income 
                        
                        fni[pred_year, agent_j] = (
                            gfi[pred_year, agent_j]
                            + SBT_c * a[pred_year, agent_j]
                            + SBT_nc
                            - 0.05 * a[pred_year, agent_j]
                            - VAT
                            - EFT
                            - 1 / ML * lt[pred_year - 1, agent_j]
                            - u1[pred_year, agent_j]
                            + rentValue
                        )

                        d[pred_year, agent_j] = (1 + R) * d[pred_year - 1, agent_j] + fni[pred_year - 1, agent_j] - NFM * MEPI + u2[pred_year, agent_j]

                        # Currrent assets
                        ca[pred_year, agent_j] = d[pred_year, agent_j]

                        # Equity   
                        e[pred_year, agent_j] = fa[pred_year, agent_j] + ca[pred_year, agent_j] - lt[pred_year, agent_j]

                        # ---- Inequation - Constraints ---- #
                        model.addConstr(d[pred_year, agent_j] >= 0, name = "Deposits >= 0")
                                                                    
                        # ---- Parameters limits - Constraints ---- #       
                        if LP_second_phase == True and pred_year == 1:
                            pass
                        else:            
                            model.addConstr(a[pred_year, agent_j] >= 0, name = "Land >= 0")
                            
                        # -- Loan adquisition
                        model.addConstr(u2[pred_year, agent_j] >= 0, name = "Loans >= 0")                       # Loans cannot be negative
                        model.addConstr(u2[pred_year, agent_j] <= 2 * e[pred_year - 1, agent_j], name = "Loans <= x Equity")            # Loans adquisition limitation

                        # -- Land buy/sell expectations
                        if LP_second_phase == True and pred_year == 1:
                            pass
                        else:      
                            model.addConstr(u1[pred_year, agent_j] <= d[pred_year - 1, agent_j], name = "U1 (buy) <= deposits")      # It is not possible to invest in land buy above the deposits capacity    
                            model.addConstr(u1[pred_year, agent_j] >= -a[pred_year - 1, agent_j], name = "U1 (sell) <= land")     # It is not possible to sell more land than available
                            model.addConstr(u1[pred_year, agent_j] <= u2[pred_year, agent_j] + 0.1 * d[pred_year - 1, agent_j], name = "u1 (buy) <= x u2") 

                        # ---- Objective Functions ---- #
                        alfa = AversionRisk ** (pred_year - c_year)      #Discount factor - high AversionRisk (riskAversion) higer importance to the future

                        # y[agent_j] = y[agent_j] + fni[pred_year, agent_j] / e[c_year - 1, agent_j]            # - Case 1
                        y[agent_j] = y[agent_j] + alfa * (fa[pred_year, agent_j] + ca[pred_year, agent_j])        # - Case 2
                        y1[agent_j] = y1[agent_j] + lt[pred_year, agent_j]


                model.Params.OutputFlag = 0  # Set to 0 to suppress output during optimization

                # Evaluate holder retirment
                holder_retire = holder_retire_ev(holder_age, holder_suc_age, holder_suc) 
                if holder_retire == 2:
                    """ 
                        If the agent aims to retire and has no successor 
                        the optimisation is simplifed to 1 year (no prediciton) 
                        and objective funcion aims to sell everything and minimise loans 
                    """
                    y[agent_j] = ca[c_year, agent_j] - lt[c_year, agent_j]
                    model.setObjective(y[0], GRB.MAXIMIZE)           

                else:
                    # REMARK ---> model.setObjectiveN MINIMISE | Priority works inverse (higher number -> higher priority) | Weight works only if same priority
                    # - Case 1 - Max FNI/e
                    # model.setObjectiveN(-y[0], index=0, priority=1, weight=0.9)
                    # model.setObjectiveN(y1[0], index=1, priority=1, weight=0.1)

                    # - Case 2 - Max TA
                    model.setObjectiveN(-y[0], index=0, priority=1, weight=0.4)
                    model.setObjectiveN(y1[0], index=1, priority=1, weight=0.6)

                model.update()
                model.optimize()

                # Update holder age
                holder_age += 1
                holder_suc_age += 1

                # Evaluate if the optimisation is feasible
                if model.Status == 4 or model.Status == 3 or model.Status == 5:
                    # logger.debug(f"Model {agents_id[agent_j]} is infeasible. Computing IIS... Attempting feasibility relaxation..")
                    # model.computeIIS()
                    # model.write("model.ilp")

                    # logger.debug(f"\nThe following constraints are in the IIS of agent {agents_id[agent_j]}:")
                    # for c in model.getConstrs():
                    #     if c.IISConstr:
                    #         logger.debug(f"{c.constrName}")

                    # logger.debug("Model INFEASIBLE. Attempting feasibility relaxation...")
                    """
                        relaxobjtype = 2  # 0: Minimize the weighted sum of relaxations. | 1: Minimize the weighted sum of squared relaxations. | 2: Minimize the weighted maximum of relaxations.
                        minrelax = True   # Minimize the relaxation
                        lbpen = False      # Allow relaxation of lower bounds
                        ubpen = True      # Allow relaxation of upper bounds
                        rhspen = True     # Allow relaxation of RHS of constraints
                    """
                
                    model.feasRelaxS(0, False, False, True)
                    model.optimize()

                    if model.Status == 4 or model.Status == 3 or model.Status == 5:
                        logger.debug(f"Agent {agents_id[agent_j]} optimisation is infesible. Optimisation failed")
                        return "error"

                # print("Model is feasible")
                # ---- Extract results ---- #
                for agent_j in range(nr_agents):
                    if LP_second_phase == True and c_year == 1:
                        a[c_year, agent_j] = a[c_year, agent_j]
                        lt[c_year, agent_j] = lt[c_year, agent_j].getValue()
                        
                        d[c_year, agent_j] = d[c_year, agent_j].getValue()
                        ca[c_year, agent_j] = ca[c_year, agent_j].getValue()

                        fa[c_year, agent_j] = fa[c_year, agent_j]
                        gfi[c_year, agent_j] = gfi[c_year, agent_j]
                        fni[c_year, agent_j] = fni[c_year, agent_j]
                        e[c_year, agent_j] = e[c_year, agent_j].getValue()
                        lr[c_year, agent_j] = ca[c_year, agent_j] / ( 1 / ML) * lt[c_year-1, agent_j]
                        # sr[c_year, agent_j] = (lt[c_year, agent_j]) / (
                        #     ca[c_year, agent_j] + fa[c_year, agent_j] - lt[c_year, agent_j]
                        #     )
                        u1[c_year, agent_j] = u1[c_year, agent_j]
                        u2[c_year, agent_j] = u2[c_year, agent_j].X

                    else:      
                        a[c_year, agent_j] = a[c_year, agent_j].getValue()
                        lt[c_year, agent_j] = lt[c_year, agent_j].getValue()
                        
                        d[c_year, agent_j] = d[c_year, agent_j].getValue()
                        ca[c_year, agent_j] = ca[c_year, agent_j].getValue()

                        fa[c_year, agent_j] = fa[c_year, agent_j].getValue()
                        gfi[c_year, agent_j] = gfi[c_year, agent_j].getValue()
                        fni[c_year, agent_j] = fni[c_year, agent_j].getValue()
                        e[c_year, agent_j] = e[c_year, agent_j].getValue()
                        lr[c_year, agent_j] = ca[c_year, agent_j] / ( 1 / ML) * lt[c_year-1, agent_j]
                        # sr[c_year, agent_j] = (lt[c_year, agent_j]) / (
                        #     ca[c_year, agent_j] + fa[c_year, agent_j] - lt[c_year, agent_j]
                        #     )
                        u1[c_year, agent_j] = u1[c_year, agent_j].X
                        u2[c_year, agent_j] = u2[c_year, agent_j].X

                    # Calculate land production (€/ha)
                    if A_va[c_year - 1, agent_j] == 0:
                        A_va[c_year - 1, agent_j] = average_ha_va

                    try:
                        u1_ha[c_year, agent_j] = u1[c_year, agent_j] / A_va[c_year - 1, agent_j]
                    except:
                        print(f"u1 = {u1[c_year, agent_j]}")
                        print(f"A_va = {A_va[c_year - 1, agent_j]}")
                    A_ha[c_year, agent_j] = A_ha[c_year - 1, agent_j] + u1_ha[c_year, agent_j]

                    if A_ha[c_year, agent_j] != 0:
                        A_va[c_year, agent_j] = a[c_year, agent_j] / A_ha[c_year, agent_j]
                        prod_ha[c_year, agent_j] = gfi[c_year, agent_j] / A_ha[c_year, agent_j]
                    else:
                        A_va[c_year, agent_j] = 0
                        prod_ha[c_year, agent_j] = 0

                if a[c_year, agent_j] == 0 and holder_retire == 2:
                    # print(f"Agents {agents_id[agent_j]} retierment")
                    break

        for v in range(c_year + 1):
            for j in range(nr_agents):
                a_f[v, j] = a[v, j]
                lt_f[v, j] = lt[v, j]
                ca_f[v, j] = ca[v, j]
                fa_f[v, j] = fa[v, j]
                d_f[v, j] = d[v, j]
                lr_f[v, j] = lr[v, j]
                sr_f[v, j] = sr[v, j]
                fni_f[v, j] = fni[v, j]
                gfi_f[v, j] = gfi[v, j]
                e_f[v, j] = e[v, j]
                np_f[v, j] = netp[v, j]
                u1_f[v, j] = u1[v, j]
                u2_f[v, j] = u2[v, j]

                a_ha_f[v, j] = A_ha[v, j]
                a_va_f[v, j] = A_va[v, j]
                prod_ha_f[v, j] = prod_ha[v, j]
                u1_ha_f[v, j] = u1_ha[v, j]


            year_f[v, j] = year + v

        # Storing the total results of the optimization in a dataframe
        agents_LP_optimisation = pd.DataFrame()
        for agent_j in range(nr_agents):
            agent_LP_opt = pd.DataFrame()
            agent_LP_opt["Land"] = a_f[:, agent_j]
            agent_LP_opt["Loans"] = lt_f[:, agent_j]
            agent_LP_opt["Current Assets"] = ca_f[:, agent_j]
            agent_LP_opt["Fixed Assets"] = fa_f[:, agent_j]
            agent_LP_opt["Deposits"] = d_f[:, agent_j]
            agent_LP_opt["Liquidity Ratio"] = lr_f[:, agent_j]
            agent_LP_opt["Solvency Ratio"] = sr_f[:, agent_j]
            agent_LP_opt["Farm Net Income"] = fni_f[:, agent_j]
            agent_LP_opt["Gross Farm Income"] = gfi_f[:, agent_j]
            agent_LP_opt["Equity"] = e_f[:, agent_j]
            agent_LP_opt["Land buy+/sell-"] = u1_f[:, agent_j]
            agent_LP_opt["Loans acquisition"] = u2_f[:, agent_j]

            agent_LP_opt["a_ha_f"] = a_ha_f[:, agent_j]
            agent_LP_opt["a_va_f"] = a_va_f[:, agent_j]
            agent_LP_opt["prod_ha_f"] = prod_ha_f[:, agent_j]
            agent_LP_opt["u1_ha_f"] = u1_ha_f[:, agent_j]

            agent_LP_opt["agent_id"] = agents_id[agent_j]
            agent_LP_opt["agent_region"] = agents_regions[agent_j]
            agent_LP_opt["Year"] = year_f[:, agent_j]

            agents_LP_optimisation = pd.concat([agents_LP_optimisation, agent_LP_opt])
        del(agent_LP_opt)

        # Release memory - del(dataframe)
        del(a_f)
        del(lt_f)
        del(ca_f)
        del(fa_f)
        del(d_f)
        del(lr_f)
        del(sr_f)
        del(fni_f)
        del(gfi_f)
        del(e_f)
        del(u1_f)
        del(u2_f)
        del(a_ha_f)
        del(a_va_f)
        del(prod_ha_f)
        del(u1_ha_f)
        del(agents_id)
        del(agents_regions)
        del(year_f)
        

    except Exception as e:
        logger.debug(f"Agent {agents_id} failed")

        # Release memory - del(dataframe)
        del(a_f)
        del(lt_f)
        del(ca_f)
        del(fa_f)
        del(d_f)
        del(lr_f)
        del(sr_f)
        del(fni_f)
        del(gfi_f)
        del(e_f)
        del(u1_f)
        del(u2_f)
        del(a_ha_f)
        del(a_va_f)
        del(prod_ha_f)
        del(u1_ha_f)
        del(agents_id)
        del(agents_regions)
        del(year_f)

        return "error"

    return agents_LP_optimisation

def match_subsidies_policies(subsidies:List[FarmYearSubsidyDTO], policies:List[PolicyJsonDTO]) -> List[dict]:
    """
        Match subsidies with policies and return a list of dictionaries with subsidy details.

        Args:
            subsidies (List[FarmYearSubsidyDTO]): A list of subsidy objects.
            policies (List[PolicyJsonDTO]): A list of policy objects.

        Returns:
            List[dict]: A list of dictionaries where each dictionary contains details about a matched subsidy.
    """
    agent_subsidies = []
    for subsidy in subsidies:
        subsidy_ID = subsidy.policyIdentifier
        for policy in policies:
            policy_ID = policy.policyIdentifier
            if subsidy_ID == policy_ID:
                subsidies_aux = {
                    "value" : subsidy.value,
                    "policyId" : policy_ID,
                    "startYear" : policy.startYearNumber,
                    "endYear" : policy.endYearNumber,
                    "isCoupled" : policy.isCoupled
                }
                agent_subsidies.append(subsidies_aux)
                break

    return agent_subsidies

def subsidies_check(agent_subsidies, year:int):
    """
        Check subsidies for a given year and calculate the total value of coupled and non-coupled subsidies.

        Args:
            agent_subsidies (List[dict]): A list of dictionaries where each dictionary contains details about subsidies.
            year (int): The year for which the subsidies are checked.

        Returns:
            tuple: A tuple containing two values:
                - The total value of coupled subsidies (SBT_c).
                - The total value of non-coupled subsidies (SBT_nc).
    """
    SBT_c = 0
    SBT_nc = 0
    for subsidy in agent_subsidies:

        if subsidy["startYear"] >= year and subsidy["endYear"] <= year:

            if subsidy["isCoupled"] == True:
                SBT_c += subsidy["value"]
            else:
                SBT_nc += subsidy["value"]

    return SBT_c, SBT_nc

def check_rent_operations(farmId: int, rentOperations: List[LandRentDTO]) -> List[dict]:
    """
        Check rent operations for a given farm and return a list of dictionaries with rent details.

        Args:
            farmId (int): The ID of the farm to check rent operations for.
            rentOperations (List[LandRentDTO]): A list of rent operation objects.

        Returns:
            List[dict]: A list of dictionaries where each dictionary contains details about rent operations related to the given farm.
    """
    landRent = []
    
    if rentOperations is None:
        rentOperations = []

    for rent_op in rentOperations:

        if rent_op.originFarmId == farmId:
            landRent_op = {
                "farmId" : farmId,
                "rentValue" : rent_op.rentValue,
                "rentArea" : rent_op.rentArea,
                "rentOut" : True
            }
            landRent.append(landRent_op)
        
        elif rent_op.destinationFarmId == farmId:
            landRent_op = {
                "farmId" : farmId,
                "rentValue" : rent_op.rentValue,
                "rentArea" : rent_op.rentArea,
                "rentOut" : True
            }
            landRent.append(landRent_op)

    return landRent

def set_algorithm_inputs(agent: ValueToLPDTO, year, agents_avg_prod_values, agents_prod, agents_ha, average_ha_va, agent_subsidies, landRent):
    """
        Prepare inputs for the algorithm.

        Args:
            agent (ValueToLPDTO): The agent data object.
            year (int): The current year.
            agents_avg_prod_values (dict): A dictionary with average production values for each agent.
            agents_prod (dict): A dictionary with production data for each agent.
            agents_ha (dict): A dictionary with total cultivated hectares for each agent.
            average_ha_va (float): Average hectare value.
            agent_subsidies (List[dict]): A list of dictionaries containing subsidy information for the agent.
            landRent (List[dict]): A list of dictionaries containing land rent information.

        Returns:
            dict: A dictionary containing prepared inputs for the algorithm.
    """
    from loguru import logger
    # Prepare inputs for the algorithm

    # -- Agent productions 
    agent_total_ha = agents_ha[agent.farmId]
    agent_avg_prod = agents_avg_prod_values[agent.farmId]["prod_per_land_value"]
    agent_avg_costs = agents_avg_prod_values[agent.farmId]["costs_per_land_value"]

    if agent.agriculturalLandValue == 0 and agent_total_ha != 0:
        # logger.debug(f"Agent {agent.farmId} a_0 (land value) = 0 and total_ha != 0. Proceed to extract land value from agent's productions.")
        agent.agriculturalLandValue = agents_prod[agent.farmId]['land_value']

    # -- Check agent initial state
    # Check deposits/TA

    if agent.sE465 == 0:
        # if agent.sE420 > 0:
        #     agent.sE465 += agent.sE420

        if agent.agriculturalLandValue == 0:
            # logger.debug(f"Agent {agent.farmId}: Land value = 0 and Total Assets = 0") 
            if agent.sE490 > 0:
                logger.debug(f"Agent {agent.farmId}: Land value = 0 | Total Assets = 0 | Loans > 0") 

    # Check equity
    equity = agent.sE465 - agent.sE490 
    if equity < 0:
        # logger.debug(f"Agent {agent.farmId} equity < 0.")
        agent.sE465 += agent.sE420

    #Land Rent
    rentArea = 0
    rentValue = 0
    if len(landRent) != 0:
        for rentOp in landRent:
            if rentOp["rentOut"] == True:
                rentArea -= rentOp["rentArea"]
                rentValue += rentOp["rentValue"]
            else:
                rentArea += rentOp["rentArea"]
                rentValue -= rentOp["rentValue"]


    algorithm_inputs = {
        "agentId" : agent.farmId,
        "Total_current_assets" : agent.sE465,
        "A_0" : agent.agriculturalLandValue, 
        "Long_loans" : agent.sE490,
        "Farm_net_income" : agent.sE420,
        "Gross_farm_income" : agent.sE410, 
        "agents_id" : agent.farmId,
        "agents_region" : agent.regionLevel3,
        "holder_age" : agent.agentHolder.holderAge, 
        "holder_succssesor" : agent.agentHolder.holderSuccessors,
        "holder_succssesor_age" : agent.agentHolder.holderSuccessorsAge,
        "land_avg_prod" : agent_avg_prod, 
        "land_avg_costs" : agent_avg_costs, 
        "NFM" : agent.agentHolder.holderFamilyMembers, 
        "SBT" : agent_subsidies, 
        "rentArea" : rentArea,
        "rentValue" : rentValue,
        "average_ha_va" : average_ha_va, 
        "AversionRisk" : agent.aversionRiskFactor, 
        "land_ha" : agent_total_ha, 
        "year" : year  
    }  
    
    return algorithm_inputs

def calculate_land_production(productions):
    """
        Calculate average production and costs values for each agent and compute average hectare value.

        Args:
            productions (List[ProductionDTO]): A list of production data objects.

        Returns:
            tuple: A tuple containing:
                - agents_avg_prod_values (dict): A dictionary with average production values per land and value for each agent.
                - agents_prod (dict): A dictionary with total production data for each agent.
                - agents_ha (dict): A dictionary with total cultivated hectares for each agent.
                - average_ha_va (float): Average hectare value.
    """
    # Extract Agents total ha
    # Calculate Agent average production (€) and costs (€) for its productions --> Use later to calculate the GFI
    agents_ha = defaultdict(float)
    agents_prod = defaultdict(lambda: {'prod_profit': 0, 'prod_costs': 0, 'land_value': 0, 'cultivated_area': 0, 'prod_count': 0})
    agents_avg_prod_values = defaultdict(lambda: {'prod_per_land_ha': 0, 'costs_per_land_ha': 0, 'prod_per_land_value': 0, 'costs_per_land_value': 0})

    for prod in productions:
        farm_id = prod.farmId    
        
        agents_ha[farm_id] += prod.cultivatedArea
        agents_prod[farm_id]['prod_profit'] += prod.valueSales 
        # agents_prod[farm_id]['prod_profit'] += prod.cropProduction #TODO: Check if cropProduction is € or tons. If €, use this.
        agents_prod[farm_id]['prod_costs'] += prod.variableCosts
        if np.isnan(prod.landValue):
            prod.landValue = 0
        agents_prod[farm_id]['land_value'] += prod.landValue
        agents_prod[farm_id]['cultivated_area'] += prod.cultivatedArea  
        agents_prod[farm_id]['prod_count'] += 1
        
    for group, data in agents_prod.items():

        prod_per_land_ha = 0
        costs_per_land_ha = 0 

        prod_per_land_value = 0
        costs_per_land_value = 0

        if data['land_value'] != 0 and data['cultivated_area'] != 0:
            prod_per_land_ha = (data['prod_profit'] / data['prod_count']) / (data['cultivated_area'] / data['prod_count'])
            costs_per_land_ha = (data['prod_costs'] / data['prod_count']) / (data['cultivated_area'] / data['prod_count'])

            prod_per_land_value = (data['prod_profit'] / data['prod_count']) / (data['land_value'] / data['prod_count'])
            costs_per_land_value = (data['prod_costs'] / data['prod_count']) / (data['land_value'] / data['prod_count'])
        
        agents_avg_prod_values[group] = {
                'prod_per_land_ha' : prod_per_land_ha,
                'costs_per_land_ha' : costs_per_land_ha,
                'prod_per_land_value' : prod_per_land_value,
                'costs_per_land_value' : costs_per_land_value
            }        
        
    # Average values of profit and costs to set as default values for agents without land
    prod_per_land_sum = 0
    costs_per_land_sum = 0

    prod_per_ha_sum = 0
    costs_per_ha_sum = 0
    
    for group in agents_avg_prod_values.keys():
        prod_per_land_sum += agents_avg_prod_values[group]["prod_per_land_value"]
        costs_per_land_sum += agents_avg_prod_values[group]["costs_per_land_value"]

        prod_per_ha_sum += agents_avg_prod_values[group]["prod_per_land_ha"]
        costs_per_ha_sum += agents_avg_prod_values[group]["costs_per_land_ha"]

    avg_profit = prod_per_land_sum / len(agents_avg_prod_values)
    avg_costs = costs_per_land_sum / len(agents_avg_prod_values)

    avg_profit_ha = prod_per_ha_sum / len(agents_avg_prod_values)
    avg_costs_ha = costs_per_ha_sum / len(agents_avg_prod_values)

    # Set default values to the avg, in case there is an agent witout productions.
    agents_avg_prod_values.default_factory = lambda: {'prod_per_land_ha': avg_profit_ha, 'costs_per_land_ha': avg_costs_ha, 'prod_per_land_value': avg_profit, 'costs_per_land_value': avg_costs, 'prod_count': 0}

    # Calculate average ha value
    ha_va = [prod.landValue/prod.cultivatedArea if prod.cultivatedArea != 0 else 0 for prod in productions]
    for i in range(len(ha_va)):
        if np.isnan(ha_va[i]):
            ha_va[i] = 0
    average_ha_va = np.mean(ha_va)

    return agents_avg_prod_values, agents_prod, agents_ha, average_ha_va

def set_LMM_inputs(productions, errors):
    """
        Extract and prepare production data for LMM (Land Market Model) analysis.

        Args:
            productions (List[ProductionDTO]): A list of production data objects.
            errors (List[int]): A list of farm IDs that failed the previous optimization.

        Returns:
            pd.DataFrame: A DataFrame with production data ready for LMM analysis.
    """
    # -- Productions extractions
    agent_id = []
    year_id = []
    land_ha = []
    land_value = []
    productionId = []
    cropProduction = []
    for production in productions:
        # Do not consider farms that failed the previous optimization
        if production.farmId not in errors:
            agent_id.append(production.farmId)
            year_id.append(production.yearId)
            land_ha.append(production.cultivatedArea)
            productionId.append(production.id)
            cropProduction.append(production.cropProduction)
            land_value.append(production.landValue)

    productions_LMM = pd.DataFrame()
    productions_LMM['agent_id'] = agent_id
    productions_LMM['year_id'] = year_id
    productions_LMM['land_ha'] = land_ha
    productions_LMM['land_value'] = land_value
    productions_LMM['ProductionId'] = productionId
    productions_LMM['cropProduction'] = cropProduction
    productions_LMM["ha_value"] = productions_LMM["land_value"] / productions_LMM["land_ha"]

    return productions_LMM

def LMM(agents_LP_decisions, productions_LMM):
    """
        Perform Land Market Model (LMM) analysis by matching buyers and sellers of land.

        Args:
            agents_LP_decisions (pd.DataFrame): A DataFrame with land purchase/sale decisions for agents.
            productions_LMM (pd.DataFrame): A DataFrame with production data for LMM analysis.

        Returns:
            tuple: A tuple containing:
                - matched_transactions (List[dict]): A list of dictionaries with details of matched land transactions.
                - land_Transactions (List[dict]): A list of dictionaries with details of land transactions, including sale price and land exchange percentage.
    """
    from loguru import logger
    # Separate buyers and sellers
    buyers = agents_LP_decisions[agents_LP_decisions['Land buy+/sell-'] > 0]
    sellers = agents_LP_decisions[agents_LP_decisions['Land buy+/sell-'] < 0]    

    # buyers = buyers[buyers["u1_ha_f"] > 0.5]
    # sellers = sellers[sellers["u1_ha_f"] < -0.5]

    buyers_c = buyers.copy()
    sellers_c = sellers.copy()

    # Match buyers and sellers
    matched_transactions = []
    land_Transactions = []

    seller_productionId = []

    # Check if there are buyers and sellers
    if buyers.shape[0] != 0 and sellers.shape[0] != 0:
        logger.debug(f"LMM. Sellers = {sellers.shape[0]} | Buyers = {buyers.shape[0]}.")
        while len(buyers_c) > 0 and len(sellers_c) > 0:
            seller_production_sold:list[tuple:(int, float)] = []

            # Match between sellers and buyers
            seller_id, buyer_id = buy_sell_match(buyers_c, sellers_c)
            
            # Extract agents
            buyer = buyers[buyers["agent_id"] == buyer_id]
            seller = sellers[sellers["agent_id"] == seller_id]

            buyer_productions = productions_LMM[productions_LMM["agent_id"] == buyer_id]
            seller_productions = productions_LMM[productions_LMM["agent_id"] == seller_id]

            buyer_offer = buyer['Land buy+/sell-'].values[0]
            seller_offer = -seller['Land buy+/sell-'].values[0]

            buyer_offer_ha = buyer['u1_ha_f'].values[0]
            seller_offer_ha = -seller['u1_ha_f'].values[0]
            # Perform transaction 
            # 1- Set the sale price and land exchanged
            if buyer_offer > seller_offer:
                transaction_price = buyer_offer
                land_exchange = min(buyer_offer_ha, seller_offer_ha) 
            else:
                transaction_price = buyer_offer
                if buyer_offer_ha < seller_offer_ha:
                    land_exchange = buyer_offer * (seller_offer_ha / seller_offer)
                else:
                    land_exchange = seller_offer_ha

            # Check if the seller has  productions
            if len(seller_productions) != 0:
                # 2.1 - Check if the seller got several productions
                if len(seller_productions) > 1:
                    land_exchange_c = land_exchange
                    seller_productions_sorted = seller_productions.sort_values("land_ha", ascending = False)

                    for _, seller_prod in seller_productions_sorted.iterrows():
                        # Production ha higher then the land to be exchanged --> Calculate the percentage of production exchanged
                        if seller_prod["land_ha"] >= land_exchange_c:
                            seller_production_sold.append((seller_prod["ProductionId"], land_exchange_c / seller_prod["land_ha"]))

                            # seller_productionId.append(seller_prod["ProductionId"])
                            # land_transfer_perc = (land_exchange / seller_prod["land_ha"])
                            break

                        # Production ha lower then the land to be exchanged --> percentage of land to be exchanged 1
                        else:
                            land_exchange_c = land_exchange_c - seller_prod["land_ha"]
                            seller_production_sold.append((seller_prod["ProductionId"], 1.0))
                            
                            # seller_productionId.append(seller_prod["ProductionId"])
                            # land_transfer_perc = 1.0
                            if land_exchange_c <= 1:
                                break

                else:
                    if land_exchange <= seller_productions['land_ha'].values[0]:
                        seller_production_sold.append((seller_productions["ProductionId"].values[0], land_exchange / seller_productions['land_ha'].values[0]))
                    else:
                        land_exchange = seller_productions['land_ha'].values[0]
                        seller_production_sold.append((seller_productions["ProductionId"].values[0], land_exchange / seller_productions['land_ha'].values[0]))

                # Drop agent
                sellers_c = sellers_c[sellers_c['agent_id'] != seller_id]
                buyers_c = buyers_c[buyers_c['agent_id'] != buyer_id]

                for seller_prod_sold in seller_production_sold:
                    prod_Id = seller_prod_sold[0]
                    land_transfer_perc = seller_prod_sold[1]

                    if land_transfer_perc > 1 or land_transfer_perc < 0:
                        logger.debug(f"Error on land percentage transfer calculation for buyer {buyer_id} and seller {seller_id}")
                    else:    
                        land_Transactions.append({
                            'productionId': prod_Id,
                            'yearId': seller_productions['year_id'].values[0],
                            'destinationFarmId': buyer_id,
                            'salePrice': transaction_price,
                            'percentage': land_transfer_perc,
                            })    
                        
                    matched_transactions.append({
                        'productionId': prod_Id,
                        'yearId': seller_productions['year_id'].values[0],
                        'Buyer_FarmId': buyer_id,
                        'Seller_FarmId': seller_id,
                        'SalePrice': transaction_price,
                        'Percentage': land_transfer_perc,
                        'Land_exchange': land_exchange
                    })     
                    

            else:
                # logger.debug(f"Agent ID {seller_id} is trying to sell {seller_offer_ha} ha with {seller_offer} €, but has no productions")
                buyers_c = buyers_c[buyers_c['agent_id'] != buyer_id]

    else: 
        if buyers.shape[0] == 0:
            logger.debug(f"No buyers. All the agents are selling land.")
        else:
            logger.debug(f"No sellers. All the agents are buying land.")

    return matched_transactions, land_Transactions 

def buy_sell_match(buyers_c, sellers_c):
    """
        Finds the optimal match between buyers and sellers based on land buy/sell values.

        The function matches buyers and sellers by identifying the pair with the smallest absolute difference
        between the buyer's buying value and the seller's selling value.

        Args:
            buyers_c (pd.DataFrame): DataFrame containing buyer information with columns 'Land buy+/sell-' and 'agent_id'.
            sellers_c (pd.DataFrame): DataFrame containing seller information with columns 'Land buy+/sell-' and 'agent_id'.

        Returns:
            tuple: A tuple containing two elements:
                - seller_id (int or None): The ID of the selected seller.
                - buyer_id (int or None): The ID of the selected buyer.
    """
    from loguru import logger

    # Sort buyers and sellers by their "Land buy+/sell-" values
    buyers_sorted = buyers_c.sort_values(by="Land buy+/sell-").reset_index(drop=True)
    sellers_sorted = sellers_c.sort_values(by="Land buy+/sell-").reset_index(drop=True)
    
    min_u1_diff = np.inf
    seller_id = buyer_id = None
    
    sellers_values = sellers_sorted["Land buy+/sell-"].values
    sellers_ids = sellers_sorted["agent_id"].values
    
    for _, buyer in buyers_sorted.iterrows():
        buyer_value = buyer["Land buy+/sell-"]
        buyer_id_temp = buyer["agent_id"]

        # Use binary search to find the closest seller value
        pos = bisect_left(sellers_values, -buyer_value)

        # Check the closest value on the left
        if pos > 0:
            left_seller_value = sellers_values[pos - 1]
            left_diff = abs(buyer_value + left_seller_value)
            if left_diff < min_u1_diff:
                min_u1_diff = left_diff
                seller_id = sellers_ids[pos - 1]
                buyer_id = buyer_id_temp

        # Check the closest value on the right (if within bounds)
        if pos < len(sellers_values):
            right_seller_value = sellers_values[pos]
            right_diff = abs(buyer_value + right_seller_value)
            if right_diff < min_u1_diff:
                min_u1_diff = right_diff
                seller_id = sellers_ids[pos]
                buyer_id = buyer_id_temp

    return seller_id, buyer_id

def holder_retire_ev(holder_age, successor_age, holder_suc):
    """
        Determines if a holder retires and whether a successor is present.

        The function assesses retirement status based on the holder's age and the presence and age of a successor.

        Args:
            holder_age (int): The age of the holder.
            successor_age (int): The age of the successor.
            holder_suc (int): Indicator if there is a successor (0 if none, non-zero otherwise).

        Returns:
            int: Retirement status of the holder:
                - 0: No retirement.
                - 1: Retirement with a successor.
                - 2: Retirement without a successor.
    """
    holder_retires = 0

    if holder_age > 64:

        if holder_suc != 0 and successor_age >= 18:
            holder_retires = 1

        else:
            holder_retires = 2

    return holder_retires

def holder_retire_get_value(holder_age, successor_age, holder_suc, target_land):
    """
        Determines if a holder retires based on additional conditions.

        The function determines the final retirement decision based on the holder's retirement status
        and whether the target land value is zero.

        Args:
            holder_age (int): The age of the holder.
            successor_age (int): The age of the successor.
            holder_suc (int): Indicator if there is a successor (0 if none, non-zero otherwise).
            target_land (int): The target land value for the holder.

        Returns:
            bool: True if the holder retires, otherwise False.
    """
    holder_retire = holder_retire_ev(holder_age, successor_age, holder_suc) 
    if holder_retire != 0:
        if holder_retire == 2 and target_land == 0:
            holder_retire = True
        elif holder_retire == 1:
            holder_retire = True
        else:
            holder_retire = False                 
    else:
        holder_retire = False
    return holder_retire

def LP_phase_1(algorithm_inputs):
    """
        Executes the Long-Term Optimization (LP) algorithm for the first phase.

        The function runs the LP algorithm for a specified simulation period and returns the results.

        Args:
            algorithm_inputs (dict): Dictionary containing the inputs required for the LP algorithm.

        Returns:
            tuple: A tuple containing:
                - result (dict): The result of the LP algorithm.
                - agentId (int): The ID of the agent for which the LP algorithm was executed.
    """
    from loguru import logger
    result = LP_algorithm(algorithm_inputs, sim_period = 10)
    return (result, algorithm_inputs["agentId"])

def LP_phase_2(algorithm_inputs, LP_second_phase = False, LMM_transactions = []):
    """
        Executes the second phase of the Long-Term Optimization (LP) algorithm.

        The function runs the LP algorithm for the second phase, considering any LMM transactions if provided.

        Args:
            algorithm_inputs (dict): Dictionary containing the inputs required for the LP algorithm.
            LP_second_phase (bool, optional): Flag indicating if this is the second phase of LP (default is False).
            LMM_transactions (list, optional): List of LMM transactions to consider in the LP algorithm (default is an empty list).

        Returns:
            tuple: A tuple containing:
                - result (dict): The result of the LP algorithm.
                - agentId (int): The ID of the agent for which the LP algorithm was executed.
                - retire (bool): Indicates if the agent retires.
    """
    from loguru import logger
    result = LP_algorithm(algorithm_inputs, sim_period = 10, LP_second_phase = LP_second_phase, LMM_transactions = LMM_transactions)
    retire = False
    if "error" not in result:
        holder_retire_get_value(algorithm_inputs["holder_age"], algorithm_inputs["holder_succssesor_age"], algorithm_inputs["holder_succssesor"], result["Land"][1])

    return (result, algorithm_inputs["agentId"], retire)

def LMM_parallel(agents_LP_LMM, productions_LMM):
    """
        Executes the Long-Medium Term Model (LMM) in parallel for given agents and productions.

        The function processes LMM for the provided agents and productions in parallel.

        Args:
            agents_LP_LMM (pd.DataFrame): DataFrame containing agent information for LMM execution.
            productions_LMM (list): List of production data relevant for LMM.

        Returns:
            tuple: A tuple containing:
                - LMM_transactions (list): List of transactions generated by the LMM.
                - land_Transactions (list): List of land transactions generated by the LMM.
    """
    from loguru import logger
    a, b =  LMM(agents_LP_LMM, productions_LMM)
    return (a,b)

def process_inputs(input: DataToLPDTO, yearNumber:int, simulationRunId = 0, use_parallel:bool = True, batch_size = 5000, dump_input = True) -> (AgroManagementDecisionFromLP):
    """
        Processes input data for agricultural land management optimization in three main phases.

        The function orchestrates the execution of the optimization model across three phases:
        - **Phase 1:** Long-Term Optimization (LP). Calculates initial results for agents.
        - **Phase 2:** Long-Medium Term Model (LMM). Executes the model to adjust land transactions.
        - **Phase 3:** Second LP Phase. Recalculates results considering LMM transactions.

        Depending on the input options, processing can be performed in parallel to enhance performance. 
        Land transactions are managed, and the output includes agricultural management decisions and errors.

        Args:
            input (DataToLPDTO): Object containing the necessary data for the process, including information on agents, agricultural productions, policies, and rent operations.
            yearNumber (int): The year of simulation for which data will be processed.
            simulationRunId (int, optional): Identifier for the simulation run, used for logging and file storage (default is 0).
            use_parallel (bool, optional): Indicates whether parallel processing should be used (default is True).
            batch_size (int, optional): Size of batches for parallel processing (default is 5000).
            dump_input (bool, optional): If True, the input data will be dumped to a JSON file for inspection (default is True).

        Returns:
            AgroManagementDecisionFromLP: Object containing the agricultural management decisions, land transactions, and a list of errors.
    """
    # Preparation of Dask for parallel execution
    cluster = None
    client = None
    if dump_input:
        with open(f"log_{simulationRunId}_{yearNumber}.json", "w") as f:
            f.write(dumps(input.dict()))

    batch_size = 5000
    
    # Read Inputs
    agents:List[ValueToLPDTO] = input.values
    productions:List[AgriculturalProductionDTO] = input.agriculturalProductions
    policies:List[PolicyJsonDTO] = input.policies
    rentOperations:List[LandRentDTO] = input.rentOperations
    year:int = yearNumber
    LT_ignore:bool = input.ignoreLP
    LMM_ignore:bool = input.ignoreLMM   # Indicates if the LMM has to be executed or not - If True, skip to Phase 3 

    # Prepare Outputs 
    outputData:List[AgroManagementDecisions]=[]
    errors:List[int] = []
    land_Transactions = []
    
    # Configure Logger
    if simulationRunId != 0:
        configure_orm_logger(settings.ORM_ENDPOINT)
    with logger.contextualize(simulationRunId=simulationRunId, logSource="long_term_worker") if simulationRunId != 0 else nullcontext():
        logger.info(f"Received request to solve LT for yearNumber: {yearNumber}. With LT_Ignore: {input.ignoreLP} and LMM_ignore: {input.ignoreLMM}")

        if not LT_ignore:
            # If LT is requested
            agents_avg_prod_values, agents_prod, agents_ha, average_ha_va = calculate_land_production(productions)

            if average_ha_va == 0:
                ha_va = [agent.agriculturalLandValue/prod.cultivatedArea for agent, prod in zip(agents, productions)]
                ha_va = [x if not np.isnan(x) else 0 for x in ha_va]
                average_ha_va = np.mean(ha_va)
            
            # Run ABM - Long Period optimisation algorithm
            # Preparing batches
            currentBatchCount = 0
            batches = list(chunks(agents, batch_size))
        
            valid_values: List[pd.DataFrame] = []
            for batch in batches:
                results = []
                if use_parallel:
                    dask.config.set(scheduler='processes', serialization='dask')
                    cluster = LocalCluster(threads_per_worker=2, n_workers=15, memory_limit = 0, dashboard_address =":8799")
                    client = Client(address=cluster)

                    for agent in batch:
                        # algorithm_inputs = set_algorithm_inputs(agent, year, agents_avg_prod_values, agents_prod, agents_ha, average_ha_va)
                        # result = dask.delayed(LP_algorithm)(algorithm_inputs, sim_period = 10)
                        # results.append((result, agent.farmId))

                        # Merge subsidies and policies
                        agent_subsidies:List[dict] = match_subsidies_policies(agent.agentSubsidies, policies)

                        #Check agent rented land
                        landRent = check_rent_operations(agent.farmId, rentOperations)
                        algorithm_inputs = set_algorithm_inputs(agent, year, agents_avg_prod_values, agents_prod, agents_ha, average_ha_va, agent_subsidies, landRent)
                        result = dask.delayed(LP_phase_1)(algorithm_inputs)
                        results.append(result)
                    results = dask.compute(*results)
                else:
                    for agent in batch:
                        start_time = dt.datetime.now()

                        # Merge subsidies and policies
                        agent_subsidies:List[dict] = match_subsidies_policies(agent.agentSubsidies, policies)

                        #Check agent rented land
                        landRent = check_rent_operations(agent.farmId, rentOperations)

                        algorithm_inputs = set_algorithm_inputs(agent, year, agents_avg_prod_values, agents_prod, agents_ha, average_ha_va, agent_subsidies, landRent)
                        result = LP_algorithm(algorithm_inputs, sim_period = 10)
                        results.append((result, agent.farmId))
                        
                errors.extend([result[1] for result in results if "error" in result[0]])
                # A list of pd.Dataframes, inlcuding the previous one and the new generated ones
                valid_values.extend([result[0] for result in results if "error" not in result[0]])
                # Which are then concatenated in a single pf.DataFrame
                currentBatchCount += 1
                logger.debug(f"LP First phase. {currentBatchCount}/{len(batches)} batches completed. {(currentBatchCount-1)*batch_size+len(batch)}/{len(agents)} agents")

                if use_parallel:
                    client.close()
                    cluster.close()

            result_df: pd.DataFrame = pd.concat(valid_values)
            del(valid_values)

            # Save Phase 1 agents' results
            agents_LP_optimisation = result_df[result_df["Year"] == year + 1]
            del(result_df)

            agents_loans_nr = len(agents_LP_optimisation[agents_LP_optimisation["Land buy+/sell-"] < 0])
            logger.debug(f"First LP run, year{year + 1}: {agents_loans_nr} agents need to contract loans")
                    
            logger.info("First phase of LP completed. Executing LMM.")
            logger.debug(f"First phase of LP completed. {len(errors)} failed.")

            LMM_transactions = [] 
            land_Transactions = []
            if LMM_ignore == False:
                #------------------- Phase 2 - Execute LMM -------------------#
                # Let's split by region3
                regions = set([x.regionLevel3 for x in agents])
                results = []
                if use_parallel:
                    dask.config.set(scheduler='processes', serialization='dask')
                    cluster = LocalCluster(threads_per_worker=2, n_workers=15, memory_limit = 0, dashboard_address =":8799")
                    client = Client(address=cluster)
                for region in regions:
                    agents_LP_LMM = agents_LP_optimisation[agents_LP_optimisation['agent_region'] == region][["agent_id","Year", "Land buy+/sell-", "u1_ha_f"]]
                    included_agents = agents_LP_LMM["agent_id"].unique()
                    these_productions = [x for x in productions if x.farmId in included_agents]
                    productions_LMM = set_LMM_inputs(these_productions, errors)
                    if use_parallel:
                        result = dask.delayed(LMM_parallel)(agents_LP_LMM, productions_LMM)
                        results.append(result)
                    else:
                        these_LMM_transactions, these_land_Transactions = LMM(agents_LP_LMM, productions_LMM)
                        LMM_transactions.extend(these_LMM_transactions)
                        land_Transactions.extend(these_land_Transactions)
                if use_parallel:
                    results = dask.compute(*results)
                    for result in results:
                        LMM_transactions.extend(result[0])
                        land_Transactions.extend(result[1])
                    client.close()
                    cluster.close()
                logger.info("LMM run completed.")
            else:
                logger.info("LMM skipped (as requested).")    

            del(agents_LP_optimisation)
                    
            #------------------- Phase 3 - Execute LP again -------------------#
            logger.info("Executing second phase of LP")    
            currentBatchCount = 0
            
            errors_phase_1 = len(errors)
            batches = list(chunks([x for x in agents if x.farmId not in errors], batch_size))
            for batch in batches:
                results = []
                
                if use_parallel:
                    dask.config.set(scheduler='processes', serialization='dask')
                    cluster = LocalCluster(threads_per_worker=2, n_workers=15, memory_limit = 0, dashboard_address =":8799")
                    client = Client(address=cluster)
                    for agent in batch:
                        # Merge subsidies and policies
                        agent_subsidies:List[dict] = match_subsidies_policies(agent.agentSubsidies, policies)

                        #Check agent rented land
                        landRent = check_rent_operations(agent.farmId, rentOperations)
                        agent_transactions = [transaction for transaction in LMM_transactions if transaction['Buyer_FarmId'] == agent.farmId or transaction['Seller_FarmId'] == agent.farmId]
                        algorithm_inputs = set_algorithm_inputs(agent, year, agents_avg_prod_values, agents_prod, agents_ha, average_ha_va, agent_subsidies, landRent)

                        result = dask.delayed(LP_phase_2)(algorithm_inputs, LP_second_phase = True, LMM_transactions = agent_transactions)
                        results.append(result)
                    results = dask.compute(*results)
                   
                else:
                    for agent in batch:

                        # Merge subsidies and policies
                        agent_subsidies:List[dict] = match_subsidies_policies(agent.agentSubsidies, policies)

                        #Check agent rented land
                        landRent = check_rent_operations(agent.farmId, rentOperations)
                        algorithm_inputs = set_algorithm_inputs(agent, year, agents_avg_prod_values, agents_prod, agents_ha, average_ha_va, agent_subsidies, landRent)
                        agent_transactions = [transaction for transaction in LMM_transactions if transaction['Buyer_FarmId'] == agent.farmId or transaction['Seller_FarmId'] == agent.farmId]

                        result = LP_algorithm(algorithm_inputs, sim_period = 10, LP_second_phase = True, LMM_transactions = agent_transactions)
                        retire = False
                        if "error" not in result:
                            holder_retire_get_value(agent.agentHolder.holderAge, agent.agentHolder.holderSuccessorsAge, agent.agentHolder.holderSuccessors, result["Land"][1])
                        results.append((result, agent.farmId, retire))
                        
                errors.extend([result[1] for result in results if "error" in result[0]])
                # A list of pd.Dataframes, inlcuding the previous one and the new generated ones
               
                for result in results:
                    if "error" not in result[0]:
                        result_dict = result[0].to_dict()
                        this_result = AgroManagementDecisions (
                            farmId = result[1],
                            yearId= year + 1, 
                            agriculturalLandArea= round(result_dict['Land'][1]/result_dict['a_va_f'][1],2) if result_dict['a_va_f'][1] != 0 else 0,
                            agriculturalLandValue= round(result_dict['Land'][1],2),
                            longAndMediumTermLoans= round(result_dict['Loans'][1],2),
                            totalCurrentAssets= round(result_dict['Deposits'][1],2),
                            averageLandValue=result_dict['a_va_f'][1],
                            targetedLandAquisitionArea= round(result_dict['u1_ha_f'][1],2), 
                            targetedLandAquisitionHectarPrice= round(result_dict['a_va_f'][1],2),
                            retireAndHandOver = result[2] )
                        outputData.append(this_result)
                    # else: 
                    #     errors.append(result[1])
                if use_parallel:
                    client.close()
                    cluster.close()
                currentBatchCount += 1
                logger.debug(f"LP Third phase. {currentBatchCount}/{len(batches)} batches completed. {(currentBatchCount-1)*batch_size+len(batch)}/{len(agents)-errors_phase_1} agents")
            logger.debug(f"Third phase of LP completed. {len(errors)} failed.")
            logger.info(f"LP-LMM run for {year + 1} completed")

        else:
            for agent in agents:
                agent_result = AgroManagementDecisions (
                    farmId = agent.farmId,
                    yearId = 0,
                    agriculturalLandArea= agent.agriculturalLandArea,
                    agriculturalLandValue = agent.agriculturalLandValue,
                    totalCurrentAssets= agent.sE465,
                    longAndMediumTermLoans= agent.sE490,
                    targetedLandAquisitionArea= 0, 
                    targetedLandAquisitionHectarPrice= 0,
                    retireAndHandOver= False,
                    averageLandValue=agent.averageHAPrice
                )
                outputData.append(agent_result)
            land_Transactions = []

    result = AgroManagementDecisionFromLP(
        agroManagementDecisions = outputData,
        landTransactions = [] if len(land_Transactions) == 0 else land_Transactions,
        errorList = errors
    )

    return result

# COMENT BEFORE MERGE 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the LP algorithm')
    parser.add_argument('--profile', action="store_true", help='Pass this argument to create a profiling file of the execution')
    parser.add_argument('--skipParallel', action="store_true", help='Pass this argument to not run the parallel test')
    parser.add_argument('--skipSequential', action="store_true", help='Pass this argument to not run the sequential test')
    args = parser.parse_args()
    
    # (a,b) = test_production_data(1148, 2015, True, True)
    (a,b) = test_production_data(1151, 2020, True, True)
    # The file that is loaded has 10000 entries, you can make shorter tests adjusting the number below
    # a.values = a.values[:500]
    # included_ids = [x.farmId for x in a.values]
    # a.agriculturalProductions = [x for x in a.agriculturalProductions if x.farmId in included_ids]
    
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()    
    
    if not args.skipParallel:
        start_time_parallel = time.time()
        result = process_inputs(a, 2020, use_parallel = True, batch_size = 400)
        stop_time_parallel = time.time()
    
    if not args.skipSequential:
        start_time_sequential = time.time()
        result = process_inputs(a, 2020, use_parallel = False, batch_size = 400)
        stop_time_sequential = time.time()
        
    if args.profile:
        pr.disable()
        pr.dump_stats("stats.prof")
    
    if not args.skipParallel: 
        print(f"Parallel time: {stop_time_parallel - start_time_parallel}")
    if not args.skipSequential:
        print(f"Sequeatial time: {stop_time_sequential - start_time_sequential}")