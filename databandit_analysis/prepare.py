from databandit import checkfailed
import h5py

# print(data)
indexed_variable_name = 'free_evolution_time'
camera = 'ProEM'

with h5py.File(data, 'r') as f:
    

    
    img = h5_file['data']['images' + camera]['Raw'][:]           
    attrs = h5_file['globals'].attrs
    indexed_variable = attrs[indexed_variable_name]

    if camera == 'ProEM':
        atoms = img[1] - img[3]
        probe = img[0] - img[2]
        
    else:
        atoms = img[0] - img[2]
        probe = img[1] - img[2]
    
    dv_atoms = atoms
    dv_probe = probe

        
        
    
#    attrs = f['results/rois_od'].attrs
#
#    df_p0 = attrs['roi_0']
#    df_p1 = attrs['roi_1']
#    df_p2 = attrs['roi_2']

    attrs = f['globals'].attrs
    df_arpfinebias = attrs['ARP_FineBias']
    df_delta_xy = attrs['delta_xy']
    df_delta_zx = attrs['delta_zx']
