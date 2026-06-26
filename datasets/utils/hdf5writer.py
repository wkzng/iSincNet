import numpy as np
import tables
import os



class H5Writer:
    def __init__(self, file_path: str, buffer_schemas: dict, mode: str = "w"):
        # buffer_schemas = {"audio": (4, 44100*4), "activity": (4,)}
        self._validate_schemas(buffer_schemas)
        self.h5 = self._init_file(file_path, mode)
        self.buffers = self._create_buffers(buffer_schemas)

    def _validate_schemas(self, schemas:dict):
        for _, shape in schemas.items():
            assert isinstance(shape, tuple) or isinstance(shape, int)

    def _init_file(self, file_path:str, mode:str):
        if mode=="w" and os.path.exists(file_path):
            os.remove(file_path)
        if mode=="a" and not os.path.exists(file_path):
            mode = "w"
        return tables.open_file(file_path, mode)

    def _create_buffers(self, schemas:dict):
        buffers = {}
        for name, shape in schemas.items():
            atom = tables.Float32Atom() if 'audio' in name else tables.Float16Atom()
            buffers[name] = self.h5.create_earray(
                where=self.h5.root, 
                name=name, 
                atom=atom, 
                shape=(0, *shape) if isinstance(shape, tuple) else (0, shape)
            )
        return buffers

    def close(self):
        """Static method to avoid circular references"""
        if self.h5 and self.h5.isopen:
            self.h5 .close()

    def write_batch(self, batch:dict):
        assert set(batch.keys()) == set(self.buffers.keys())
        for name, data in batch.items():
            data = np.expand_dims(data, axis=0)
            self.buffers[name].append(data)
