import databandit as db

# params = {'date': '20161214',
#           'devices':  ['Flea3', 'SC1'],
#           'filerange': [960, 970]}
#
# ds = db.DataSequence('configurations/setlist/setlist.py', **params)
# ds.filesfromftp()
#
# # ds.prepare(redo=True)
# # ds.analyse(redo=True)
# # ds.postprocess()
# # print(ds.dataframe)
# print()


params = {'date': '20170523',
          'experiment': 'RF_highpower_XYZ',
          'index': 11,
          'shots': 30}

ds = db.DataSequence('configurations/labscript.py', **params)
ds.filesfromftp(onlydiff=True)

ds.prepare(redo=True)
ds.analyse(redo=True)
ds.postprocess()
print(ds.dataframe)
