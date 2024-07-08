from datetime import datetime
import shutil
from pathlib import Path
import yaml
import sys, os
sys.path.append(str((Path(os.getcwd()) / 'external').resolve()))

from assets.garment_programs.meta_garment import MetaGarment
from assets.body_measurments.body_params import BodyParameters
from external.customconfig import Properties


if __name__ == '__main__':

    # 
    # body_file = './assets/body_measurments/f_smpl_avg.yaml'
    # body_file = './assets/default_bodies/mean_all.yaml'
    # body_file = r"C:\Users\MariaKo\Documents\Logs\datasets\gen_errors\rand_NRXDPYOARD\body_measurements.yaml"
    # body_file = r"C:\Users\MariaKo\Documents\Logs\datasets\gen_errors\rand_2RVZ9O8V6J\body_measurements.yaml"
    body_file = r"C:\Users\MariaKo\Documents\Logs\datasets\gen_errors\rand_160S53YNAF\body_measurements.yaml"
    # body_file = r"G:\My Drive\GarmentCode\sewing_siggraph_garment\measurements\Maria.yaml"
    # body_file = './assets/body_measurments/f_avatar.yaml'
    # body_file = './assets/body_measurments/f_smpl_model.yaml'
    # body_file = './assets/body_measurments/f_smpl_model_fluffy.yaml'
    # body_file = './assets/body_measurments/m_smpl_avg.yaml'
    # body_file = r"C:\Users\mariako\Documents\Data\GarmentCodeData\garment-first-samples\measurements\00013.yaml"

    body = BodyParameters(body_file)

    design_files = {
        # 'base': './assets/design_params/base.yaml',
        # 'gen_err': r"C:\Users\MariaKo\Documents\Logs\datasets\gen_errors\rand_2RVZ9O8V6J\design_params.yaml"
        'gen_err_3': r"C:\Users\MariaKo\Documents\Logs\datasets\gen_errors\rand_160S53YNAF\design_params_edit.yaml"
        # 'gen_err_2': r"C:\Users\MariaKo\Documents\Logs\datasets\gen_errors\rand_NRXDPYOARD\design_params.yaml"
        # 'thin_skirt': r'c:\Users\mariako\Documents\Logs\datasets_gen\to_compare_garments_5000_0_regen_240613-14-58-48_errors\default_body\data\rand_N0ASZSMN5U\design_params_edited.yaml'
        # 'mp_skirt_flip': r"C:\Users\mariako\Documents\Logs\datasets_gen\to_compare_garments_5000_0_regen_240613-14-58-48_errors\default_body\data\rand_BSH8MAZ3S9\design_params_edited.yaml"
        # 'st_err': r"C:\Users\mariako\Documents\Logs\datasets_gen\to_compare_garments_5000_0_regen_240613-14-58-48_errors\default_body\data\rand_A35G0NQPF9\design_params.yaml"
        # 'sleeve_align': './assets/design_params/sleeve_alignment_1.yaml',
        # 'sleeve_long': './assets/design_params/sleeve_too_long.yaml',
        # 'pants_align': './assets/design_params/pants_alignment.yaml',
        # 'js_base': './assets/design_params/jumpsuit_fit.yaml',
        # 'our_dress': r"G:\My Drive\GarmentCode\sewing_siggraph_garment\designs\dress_design_params.yaml"
        # 'jing_dress': r"C:\Users\mariako\Documents\Code\Procedural-Garments\Logs\Configured_design__240117-19-06-53\design_params_manual.yaml"
        # 'debug': './Logs/Configured_design__231117-17-17-27/design_params.yaml' 
        # 'default': './assets/design_params/default.yaml',
        # 'modern': './assets/design_params/modern.yaml',
        # 'Dress_20s': './assets/design_params/dress_20s.yaml',
        # 'Dress_30s': './assets/design_params/dress_30s_header.yaml',
        # 'Dress_40s': './assets/design_params/dress_40s.yaml',
        # 'Dress_50s': './assets/design_params/dress_50s.yaml',
        # 'Dress_regency': './assets/design_params/dress_regency.yaml',
        # 'sweatshirt': './assets/design_params/sweatshirt.yaml',
        # # 'pants': './assets/design_params/pants.yaml',
        # 'jumpsuit': './assets/design_params/jumpsuit.yaml',
    }
    designs = {}
    for df in design_files:
        with open(design_files[df], 'r') as f:
            designs[df] = yaml.safe_load(f)['design']
    
    test_garments = [
        # WB(),
        # CuffBand('test', design['pants']),
        # CuffSkirt('test', design['pants']),
        # CuffBandSkirt('test', design['pants'])
    ]
    for df in designs:
        test_garments.append(MetaGarment(df, body, designs[df]))
        try:
            test_garments[-1].assert_total_length()
        except BaseException as e:
            print(e)
            pass

    for piece in test_garments:
        pattern = piece.assembly()

        if piece.is_self_intersecting():
            print(f'{piece.name} is Self-intersecting')

        # Save as json file
        sys_props = Properties('./system.json')
        folder = pattern.serialize(
            # Path(r"G:\My Drive\GarmentCode\sewing_siggraph_garment\Fits"),  #   
            Path(sys_props['output']), 
            tag='_' + datetime.now().strftime("%y%m%d-%H-%M-%S"), 
            to_subfolder=False, 
            with_3d=False, with_text=False, view_ids=False)

        body.save(folder)
        if piece.name in design_files:
            shutil.copy(design_files[piece.name], folder)
        else:
            shutil.copy(design_files['base'], folder)

        print(f'Success! {piece.name} saved to {folder}')
