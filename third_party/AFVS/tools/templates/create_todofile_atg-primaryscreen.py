#!/usr/bin/env python
# coding: utf-8

# Arguments
# 1: Path to the dimension averaged tranche scores file (if tranche scoring mode == dimension_averaging) or path to the prescreening ranking subset-1 file (if tranche_scoring_mode is tranche_min_score or tranche_ave_score)
# 2: Path to collections file of complete library in parquet format (Enamine_REAL_Space_2022q12.collections.parquet)
# 3: Path to tranches file of complete library in parquet format (Enamine_REAL_Space_2022q12.tranches.parquet)
# 4: Tranche scoring mode: dimension_averaging, tranche_min_score or tranche_ave_score
# 5: Output filename of local todo-file that will be generated
# 6: Target number of ligands for primary screen

# Imports and Settings
import pandas as pd
import copy
import os
import sys
from IPython.display import display, HTML
import re


# Arguments
print("Loading library collections file ...")
df_library_collections = pd.read_parquet(sys.argv[2])
print("Loading library tranches file...")
df_library_tranches = pd.read_parquet(sys.argv[3])
tranche_scoring_mode = sys.argv[4]
ligand_count_aim = int(os.fsencode(sys.argv[6]))
tranche_filter_regex = sys.argv[7]

if tranche_scoring_mode == "dimension_averaging":

    # Enamine_REAL_Space_2022q12_sparse.tranche-scores.csv
    df_tranche_scores = pd.read_csv(sys.argv[1])
    df_tranche_scores = df_tranche_scores.dropna()
    df_tranche_scores = df_tranche_scores.reset_index(drop=True)

    # Getting the minimum scores for each tranche
    for n in range(0, 18):
        df_tranche_scores.loc[df_tranche_scores['Tranche'] == n,'Min'] = df_tranche_scores.loc[df_tranche_scores['Tranche'] == n].min()['Score']

    # Getting the score difference relative to the minimum in each tranche
    df_tranche_scores['Diff'] = df_tranche_scores['Score'] - df_tranche_scores['Min']

    # Getting the initial tranche classes for each tranche
    regex = ''
    selected_classes = list(df_tranche_scores.loc[df_tranche_scores['Diff'] == 0, 'Class'])

    # Dropping the uses tranche classes
    df_tranche_scores = df_tranche_scores.drop(df_tranche_scores[df_tranche_scores.Diff == 0].index)


    # Forming regex
    for Tranche in range(0, 18):
            regex = regex + '[' + selected_classes[Tranche] + ']'

    # Counting the ligands
    ligands_selected = df_library_collections[df_library_collections.Tranche.str.contains(regex)]['LigandCount'].sum()

    # Adding additional tranche classes

    # Loop until we have enough ligands
    # while ligands_selected < ligand_count_aim:
    #
    #     # Finding the minimum difference tranche class
    #     selected_class = df_tranche_scores[df_tranche_scores.Diff == df_tranche_scores.Diff.min()].head(1)
    #     #selected_class = df_tranche_scores[df_tranche_scores.Score == df_tranche_scores.Score.min()].head(1)
    #
    #     # Dropping the selected tranche class
    #     df_tranche_scores = df_tranche_scores.drop(selected_class.index)
    #
    #     # Re-indexing
    #     selected_class = selected_class.reset_index(drop=True)
    #     df_tranche_scores = df_tranche_scores.reset_index(drop=True)
    #
    #     # Adding the new tranche class to regex of selected tranches
    #     added_classes = copy.deepcopy(selected_classes)
    #     selected_classes[selected_class.at[0,'Tranche']] = selected_classes[selected_class.at[0,'Tranche']] + selected_class.at[0,'Class']
    #     added_classes[selected_class.at[0,'Tranche']] = selected_class.at[0,'Class']
    #
    #     # Reforming regex strings
    #     regex_previous = copy.deepcopy(regex)
    #     regex_added = ''
    #     regex = ''
    #     for Tranche in range(0, 18):
    #         regex = regex + '[' + selected_classes[Tranche] + ']'
    #     for Tranche in range(0, 18):
    #         regex_added = regex_added + '[' + added_classes[Tranche] + ']'
    #
    #     # Calculating the number of ligands selected
    #     ligands_selected_previous = ligands_selected
    #     ligands_selected = df_library_tranches[df_library_tranches.Tranche.str.contains(regex)]['LigandCount'].sum()
    #
    #     # Printing status
    #     print("")
    #     print("Previous tranche regex string: ", regex_previous)
    #     print("Added tranche regex string:    ", regex_added)
    #     print("Current tranche regex string:  ", regex)
    #     print("Current number of ligands:     ", ligands_selected)
    # Loop until we have enough ligands
    while ligands_selected < ligand_count_aim:
        
        # Finding the minimum difference tranche class
        selected_class = df_tranche_scores[df_tranche_scores.Diff == df_tranche_scores.Diff.min()].head(1)
        
        # Resetting indices for easy access
        selected_class = selected_class.reset_index(drop=True)
        df_tranche_scores = df_tranche_scores.reset_index(drop=True)
        
        # Get the tranche value for regex matching
        tranche_to_add = selected_class.at[0, 'Tranche']
        
        # Checking regex match with tranche_filter_regex
        if re.fullmatch(tranche_filter_regex, str(tranche_to_add)):
            print(f"Tranche {tranche_to_add} matches the tranche_filter_regex. Including tranche...")
            
            # Dropping the selected tranche class
            df_tranche_scores = df_tranche_scores.drop(selected_class.index)
            
            # Adding the new tranche class to regex of selected tranches
            added_classes = copy.deepcopy(selected_classes)
            selected_classes[tranche_to_add] += selected_class.at[0, 'Class']
            added_classes[tranche_to_add] = selected_class.at[0, 'Class']
            
            # Reforming regex strings
            regex_previous = copy.deepcopy(regex)
            regex_added = ''
            regex = ''
            for Tranche in range(0, 18):
                regex += '[' + selected_classes[Tranche] + ']'
            for Tranche in range(0, 18):
                regex_added += '[' + added_classes[Tranche] + ']'
            
            # Recalculate the number of ligands selected
            ligands_selected_previous = ligands_selected
            ligands_selected = df_library_tranches[df_library_tranches.Tranche.str.contains(regex)]['LigandCount'].sum()
            
            # Print status
            print("")
            print("Previous tranche regex string: ", regex_previous)
            print("Added tranche regex string:    ", regex_added)
            print("Current tranche regex string:  ", regex)
            print("Current number of ligands:     ", ligands_selected)
        else:
            print(f"Tranche {tranche_to_add} does not match the tranche_filter_regex. Skipping tranche...")
            # Drop the unmatched tranche without adding it to the regex
            df_tranche_scores = df_tranche_scores.drop(selected_class.index)
        
    # electing subset of the last added tranche to prevent overselection
    print("Selecting subset of the last added tranche to prevent overselection ...")

    # Copying the relevant rows
    df_ERS_todo_previous = df_library_collections[df_library_collections.Tranche.str.contains(regex_previous)].copy()
    df_ERS_todo_added = df_library_collections[df_library_collections.Tranche.str.contains(regex_added)].copy()

    # Loop for adding ligands
    index = 0
    ligands_selected_current = ligands_selected_previous
    df_ERS_todo_added['Selected'] = "no"
    for ind in df_ERS_todo_added.index:

        # Adding ligands for next collection
        ligands_selected_current = ligands_selected_current + df_ERS_todo_added['LigandCount'][ind]

        # Marking collected as selected
        df_ERS_todo_added['Selected'][ind] = "yes"
        print("ligands_selected_current: ", ligands_selected_current)
        if ligands_selected_current >= ligand_count_aim:
            print("Reached target number of ligands: ", ligands_selected_current)
            print()
            break

    # Saving to output file
    print("Saving new todo file")
    df_ERS_todo_previous["Collection"] = df_ERS_todo_previous["Collection"].astype(str)
    df_ERS_todo_previous['Collection'] = df_ERS_todo_previous['Collection'].str.zfill(7)
    df_ERS_todo_previous['Collection'] = df_ERS_todo_previous[['Tranche', 'Collection']].agg('_'.join, axis=1)
    df_ERS_todo_added["Collection"] = df_ERS_todo_added["Collection"].astype(str)
    df_ERS_todo_added['Collection'] = df_ERS_todo_added['Collection'].str.zfill(7)
    df_ERS_todo_added['Collection'] = df_ERS_todo_added[['Tranche', 'Collection']].agg('_'.join, axis=1)
    df_ERS_todo_previous[["Collection", "LigandCount"]].to_csv(sys.argv[5],index=False,header=False,sep=' ')
    df_ERS_todo_added[df_ERS_todo_added.Selected == "yes"][["Collection", "LigandCount"]].to_csv(sys.argv[5],index=False,mode='a',header=False,sep=' ')

elif tranche_scoring_mode in ["tranche_min_score", "tranche_ave_score"]:

    # Settings
    pd.set_option('display.max_rows', 10)
    display(HTML("<style>.container { width:100% !important; }</style>"))

    # Loading the data from files
    print("Loading prescreening docking score file ...")
    df_prescreen_scores = pd.read_csv(sys.argv[1], usecols=[0, 3], compression='gzip')

    # Calculating the scores for each tranche
    print("Calculating scores for each tranche to obtain activity map ...")
    tranche_score_mode = "min_score"
    if tranche_scoring_mode == "tranche_min_score":
        # Calculate minimum score for each Tranche in df_library_tranches
        tranche_scores = df_prescreen_scores.groupby('Tranche')['ScoreMin'].min().reset_index()
        tranche_scores.rename(columns={'ScoreMin': 'TrancheScore'}, inplace=True)
    elif tranche_scoring_mode == "tranche_ave_score":
        # Calculate average score for each Tranche in df_library_tranches
        tranche_scores = df_prescreen_scores.groupby('Tranche')['ScoreMin'].mean().reset_index()
        tranche_scores.rename(columns={'ScoreMin': 'TrancheScore'}, inplace=True)
    else:
        # Handle error: invalid mode
        raise ValueError("Invalid tranche_score_mode. Please specify either 'tranche_min_score' or 'tranche_ave_score'.")

    # Merging tranche_scores to df_library_collections
    df_library_collections = pd.merge(df_library_collections, tranche_scores, on='Tranche', how='left')

    # Sorting by score in ascending order
    df_library_collections.sort_values(by='TrancheScore', inplace=True)

    # Selecting the ligand collections for the ATG Primary Screen
    # Variables
    i = 0
    ligands_selected = 0
    ligands_selected_next = 0

    # Adding new column to df_library_collections
    df_library_collections.loc[:, 'Selected'] = False

    # Loop until we have enough ligands
    print("Selecting ligand collections for ATG Primary Screen ...")
    print("")
    while ligands_selected < ligand_count_aim:
        
        # Checking if next ligand collection matches regex
        print("")
        print("Considering next ligand collection")
        print("tranche_filter_regex: ", tranche_filter_regex)
        print("Current ligand collection: ", df_library_collections.iloc[i].Tranche)
        print("Checking regex match...")
        if re.fullmatch(tranche_filter_regex, df_library_collections.iloc[i].Tranche):
            print("The tranche matches the tranche_filter_regex. Including tranche...")
            
            # Selecting next ligand collection
            df_library_collections.iloc[i, df_library_collections.columns.get_loc('Selected')] = True
            print("Ligand collection selected:")
            print(df_library_collections.iloc[i])
            
            # Calculating the number of ligands selected
            ligands_selected = df_library_collections.loc[df_library_collections["Selected"] == True, "LigandCount"].sum()
            print("Ligands selected: ", ligands_selected)
            ligands_considered_total = df_library_collections.iloc[0:i+1]["LigandCount"].sum()
            print("Percentage of ligands accepated so far: ", (ligands_selected/ligands_considered_total)*100,"%")
            
            # Printing status
            print("Total number of ligands selected: ", ligands_selected)
            
        else:
            print("The tranche does not match the tranche_filter_regex. Skipping tranche...")
        
        # Index
        i = i + 1

    # Create filtered dataframe
    print()
    print("Preparing new todo file...")
    filtered_df = df_library_collections[df_library_collections['Selected'] == True].copy()

    # Create new column TrancheCollection
    filtered_df.loc[:, 'TrancheCollection'] = filtered_df[['Tranche', 'Collection']].agg('_'.join, axis=1)
    # using a lambda function might be faster
    # filtered_df.loc[:, 'TrancheCollection'] = filtered_df.apply(lambda row: f"{row['Tranche']}_{row['Collection']}", axis=1)

    # Save to CSV
    print("Saving new todo file ..." + sys.argv[5])
    filtered_df[['TrancheCollection', 'LigandCount']].to_csv(sys.argv[5], index=False, header=False, sep=' ')
    print("")

else:
    # Handle error: invalid mode
    raise ValueError("Invalid tranche_score_mode. Please specify either 'dimension_averaging', 'tranche_min_score' or 'tranche_ave_score'.")