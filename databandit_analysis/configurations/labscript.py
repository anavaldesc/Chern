from datetime import datetime
import os
from pathlib import Path

# if databandit not in $PYTHONPATH
# import sys
# sys.path.append('/Users/dimitris/Desktop/databandit')
from databandit import Config


class LabScript(Config):
    def __init__(self, **kwargs):
        # TODO documentation

        # full remote_path is Labscript_Data/Experiments/RbLi

        self.experiment = kwargs['experiment']
        self.index = kwargs['index']
        self.shots = kwargs['shots']

        self.date = datetime.strptime(kwargs['date'], '%Y%m%d')
        datapath_parts = self.date.strftime('%Y %m %d').split()

        p = Path('./Labscript_Data/Experiments/RbLi')
        self.folderpath = p.joinpath(self.experiment,
                                     *datapath_parts,
                                     f'{self.index:04}')

    def filenames(self):

        # if self.folderpath.exists():
        #     files = self.folderpath.glob('*.h5')
        # else:  # make the filenames if no local folder
        date = self.date.strftime('%Y_%m_%d')
        files = []
        for shot in range(self.shots):
            file = Path(f'{date}_{self.experiment}_{shot:02}.h5')
            files.append(self.folderpath / file)

        return list(map(str, files))

    def outputfile(self):
        return self.experiment + self.date.strftime('_%d%b%Y_') + \
            f'{self.index:04}.h5'
