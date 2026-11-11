"""Validate saved browser ZIP artifacts with zipfile, h5py and native pyMTEX 1.0.0.

Usage: python web/tests/check-download-zips.py /path/printed/by/batch-downloads-browser.mjs
ANG import: grid=False, setting=2; H5OINA uses its native importer/default frame.
Units and frame conversion are delegated to the unchanged native export/import paths.
"""
import io,json,sys,tempfile,zipfile
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from mtex import loadEBSD
folder=Path(sys.argv[1]);archives=sorted(folder.glob('*.zip'))
assert archives,'No browser downloads found'
for path in archives:
    stem=path.stem
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None,path
        entries=archive.namelist()
        vendor='.h5oina' if any(name.endswith('.h5oina') for name in entries) else '.ang'
        assert sorted(entries)==sorted(f'{stem}/{stem}{suffix}' for suffix in ('.h5',vendor,'_IPF-Z.png','_NCC.png','.json')),(path,entries)
        assert all(info.compress_type==zipfile.ZIP_STORED for info in archive.infolist())
        project=json.loads(archive.read(f'{stem}/{stem}.json'))
        assert project['batch']['transport']=='downloads' and project['batch']['status']=='archive-ready'
        assert len(project['batch']['outputs'])==5 and all(entry['complete'] for entry in project['batch']['outputs'])
        rows,cols=project['scan_info']['rows'],project['scan_info']['cols']
        with h5py.File(io.BytesIO(archive.read(f'{stem}/{stem}.h5'))) as result:
            assert result['zebsdsolve/Maps/IPFZ_RGBA'].shape==(rows,cols,4)
            assert result['zebsdsolve/Maps/NCC'].shape==(rows,cols)
            for suffix in ('_IPF-Z.png','_NCC.png'):
                image=Image.open(io.BytesIO(archive.read(f'{stem}/{stem}{suffix}'))).convert('RGBA')
                assert image.size==(cols,rows)
                key='IPFZ_RGBA' if suffix=='_IPF-Z.png' else 'NCC_RGBA'
                np.testing.assert_array_equal(np.asarray(image),result['zebsdsolve/Maps/'+key][()])
        vendor_bytes=archive.read(f'{stem}/{stem}{vendor}')
        with tempfile.TemporaryDirectory(prefix='zebsdsolve-downloaded-import-') as temporary:
            vendor_path=Path(temporary)/('result'+vendor);vendor_path.write_bytes(vendor_bytes)
            options={'grid':False}
            if vendor=='.ang':options['setting']=2
            ebsd=loadEBSD(str(vendor_path),**options)
            assert len(ebsd.x)==rows*cols,(path,len(ebsd.x),rows*cols)
        if vendor=='.h5oina':
            reference=json.loads((folder/'p6-reference.json').read_text())
            with h5py.File(io.BytesIO(vendor_bytes)) as result:
                root=next(group for group in result.values() if isinstance(group,h5py.Group) and 'EBSD' in group)
                data=root['EBSD/Data']
                assert 'Processed Patterns' not in data and 'Unprocessed Patterns' not in data
                np.testing.assert_allclose(data['Euler'][reference['cell']],reference['euler'],atol=1e-6,rtol=0)
    print(path.name,': CRC, 5 matching members, native maps, JSON state and pyMTEX import passed')
print('All browser ZIP exports passed independent Python zipfile/h5py and native pyMTEX validation.')
