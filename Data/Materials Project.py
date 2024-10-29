from mp_api.client import MPRester

with MPRester("your_api_key") as mpr:
    docs = mpr.summary.search(formula="ABX3", fields=['formula_pretty', 'band_gap', 'formation_energy_per_atom', 'structure', 'material_id'])
    mpid_dict = {doc.material_id: [doc.formula_pretty, doc.band_gap, doc.formation_energy_per_atom, doc.structure] for doc in docs}

formulas = []
band_gaps = []
form_en = []
ids = []
struct = []
phases = []
for key in mpid_dict:
        ids.append(key)
        formulas.append(mpid_dict[key][0])
        band_gaps.append(mpid_dict[key][1])
        form_en.append(mpid_dict[key][2])
        struct.append(mpid_dict[key][3])

halides = {}
for a in range(len(formulas)):
    if formulas[a].endswith("Cl3") or formulas[a].endswith("Br3") or formulas[a].endswith("F3") or formulas[a].endswith("I3"):
            halides[a] = formulas[a]

en_hal = {}
for i in halides:
    en_hal[i] = form_en[i]

elements = {}
import re
for a in halides:
    x = re.findall('[A-Z][^A-Z]*', halides[a])
    x[2] = x[2][:-1]
    elements[a] = x

final_elements = {}
original_indexes = []
for i in elements:
    a = elements[i][0]
    b = elements[i][1]
    if a not in ['Cl', 'Br', 'F', 'I', 'N', 'O', 'P', 'S', 'U', 'H'] and b not in ['Cl', 'Br', 'F', 'I', 'N', 'O', 'P', 'S', 'U', 'H']:
        original_indexes.append(i)
        final_elements[i] = elements[i]

import pandas as pd
el_prop = pd.read_excel("Data/Elemental_properties.xlsx")
el_prop.set_index("M", inplace=True)
props = el_prop.columns.to_list()

def swapList(sl,pos1,pos2):
    n = len(sl)     
    temp = sl[pos1]
    sl[pos1] = sl[pos2]
    sl[pos2] = temp
    return sl 

for i in final_elements:
        if el_prop.loc[final_elements[i][0], "Ion_rad"] < el_prop.loc[final_elements[i][1], "Ion_rad"]:
            final_elements[i] = swapList(final_elements[i], 0, 1)

decomps = []
for a in final_elements:
    ax = final_elements[a][0] + final_elements[a][2]
    bx2 = final_elements[a][1] + final_elements[a][2] + '2'
    decomps.append([ax, bx2])

valid_perovskites = []
for a in range(len(decomps)):
    try:
        with MPRester("65Rscm2pe2NukF9C2GbJgbo8g89KcSZr") as mpr:
            docs = mpr.summary.search(formula = decomps[a][0], fields=['formation_energy_per_atom', 'material_id'])
            mpid_dict = {doc.material_id: [doc.formation_energy_per_atom] for doc in docs}

        with MPRester("65Rscm2pe2NukF9C2GbJgbo8g89KcSZr") as mpr:
            docs = mpr.summary.search(formula = decomps[a][1], fields=['formation_energy_per_atom', 'material_id'])
            mpid_dict1 = {doc.material_id: [doc.formation_energy_per_atom] for doc in docs}
        
        if len(mpid_dict) != 0 and len(mpid_dict1) != 0:
            valid_perovskites.append([original_indexes[a], mpid_dict, mpid_dict1])
    
    except:
        pass

decomp_energies = {}
for i in valid_perovskites:
    abx3 = en_hal[i[0]]
    ax = min(i[1].values())[0]
    bx2 = min(i[2].values())[0]
    en = abx3*5 - ax*2 - bx2*3
    decomp_energies[i[0]] = en

formulas_final = []
band_gaps_final = []
form_en_final = []
ids_final = []
struct_final = []
decomposition_energies = []
for ind in decomp_energies:
        ids_final.append(ids[ind])
        formulas_final.append(formulas[ind])
        band_gaps_final.append(band_gaps[ind])
        form_en_final.append(form_en[ind])
        struct_final.append(struct[ind])
        decomposition_energies.append(decomp_energies[ind])

phase_dict = {}
for i in range(len(ids_final)):
    with MPRester("65Rscm2pe2NukF9C2GbJgbo8g89KcSZr") as mpr:
        docs = mpr.summary.search(material_ids = [ids_final[i]], fields=["symmetry"])
    phase_dict[ids_final[i]] = [str(doc.symmetry.crystal_system) for doc in docs]

phases = []
for key in phase_dict:
    p = phase_dict[key]
    phases.append(p[0])

mp_data = pd.DataFrame(list(zip(ids_final, formulas_final, phases, decomposition_energies, band_gaps_final, form_en_final, struct_final)),
               columns =["MP_id", 'Formula', 'Phase', "Decomposition Energy", 'Band Gap', 'Formation Energy per atom', "Structure"])

mp_data["Formation Energy per atom"] = pd.to_numeric(mp_data["Formation Energy per atom"])
min_energy_idx = mp_data.groupby(['Formula', 'Phase'])['Formation Energy per atom'].idxmin()
mp_data = mp_data.loc[min_energy_idx]