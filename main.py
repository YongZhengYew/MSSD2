import numpy as np

# np datatypes
jindex = np.dtype([("s", np.int64), ("p", np.int64), ("e", np.bool)])
cg = np.dtype([("x", np.longdouble), ("y", np.longdouble), ("z", np.longdouble), ("e", np.bool)])

def cg_to_r(_cg):
    return np.sqrt((_cg["x"])**2 + (_cg["y"])**2 + (_cg["z"])**2)

def mul_cg_mass(_cg, _mass):
    if not _cg["e"]:
        return (-99, -99, -99, False)
    else:
        return (_cg["x"] * _mass, _cg["y"] * _mass, _cg["z"] * _mass, True)

def add_cgs(_cg1, _cg2):
    if not _cg1["e"]:
        return _cg2
    elif not _cg2["e"]:
        return _cg1
    else:
        return (_cg1["x"] + _cg2["x"], _cg1["y"] + _cg2["y"], _cg1["z"] + _cg2["z"], True)

# Global values
g_pile_height = 8
g_blocks_in_row = 3
assert(g_blocks_in_row % 2 == 1)
g_mid_row = g_blocks_in_row//2

g_l = 42
g_w = 14
g_h = 8
g_mass = 1

# Init entire jenga pile in jindex
class Pile():
    def __init__(self, _pile_height, _blocks_in_row, _l, _w, _h, _mass):
        self.pile_height = _pile_height
        self.blocks_in_row = _blocks_in_row

        self.l = _l
        self.w = _w
        self.h = _h
        self.mass = _mass
        assert(self.blocks_in_row // 2 == 1)
        self.mid_row = self.blocks_in_row//2

        self.jindex_pile = np.zeros((self.pile_height, self.blocks_in_row), dtype=jindex)
        for i in range(self.pile_height):
            for j in range(self.blocks_in_row):
                self.jindex_pile[i, j] = (i, j - self.mid_row, True)
        self.refresh_cg_pile()
        self.create_sub_pile(0)
        
    def refresh_cg_pile(self):
        self.cg_pile = np.zeros((self.pile_height, self.blocks_in_row), dtype=cg)
        for i in range(self.pile_height):
            for j in range(self.blocks_in_row):
                self.cg_pile[i, j] = self.jindex_to_cg(self.jindex_pile[i, j])
    
    def jindex_to_cg(self, jdex):
        if jdex["e"]:
            is_even = jdex["s"] % 2 == 0
            return (jdex["p"]*self.w if is_even else 0, 0 if is_even else jdex["p"]*self.w, jdex["s"]*self.h + self.h/2, True)
        else:
            return(-99, -99, -99, False)

    def create_sub_pile(self, start_s):
        self.jindex_sub_pile = np.zeros((self.pile_height - start_s, self.blocks_in_row), dtype=jindex)
        for i in range(start_s, self.pile_height):
            for j in range(self.blocks_in_row):
                self.jindex_sub_pile[i-start_s,j] = self.jindex_pile[i,j]
                self.jindex_sub_pile[i-start_s,j]["s"] -= start_s

        self.cg_sub_pile = np.zeros((self.pile_height - start_s, self.blocks_in_row), dtype=cg)
        for i in range(len(self.jindex_sub_pile)):
            for j in range(self.blocks_in_row):
                self.cg_sub_pile[i, j] = self.jindex_to_cg(self.jindex_sub_pile[i, j])

    
    def sum_sub_pile_cg_mass(self):
        total = np.array([(0,0,0,True)],dtype=cg)
        reg_mul = np.array([(0,0,0,True)],dtype=cg)
        for row in self.cg_sub_pile:
            for cg_inst in row:
                reg_mul[0] = mul_cg_mass(cg_inst, self.mass)
                total[0] = add_cgs(total[0], reg_mul[0])
        return total[0]
    
    def sum_sub_pile_mass(self):
        total = 0
        for row in self.jindex_sub_pile:
            for cg_inst in row:
                if cg_inst["e"]:
                    total += self.mass
        return total
    
    def get_sub_pile_cg(self):
        res_sum_sub_pile_cg_mass = np.array([self.sum_sub_pile_cg_mass()],dtype=cg)
        res = np.array([(0,0,0,True)],dtype=cg)
        res[0] = mul_cg_mass(res_sum_sub_pile_cg_mass[0], 1/self.sum_sub_pile_mass())
        return res[0]
    
    def get_sub_pile_base_k_coeff(self):
        _k_directory = {
            (0,3): 1,
            (0,2): 1,
            (0,1): 3,
            (1,2): 2,
            (-1,2): 2,
            (1,1): 4,
            (-1,1): 4,
            (0,0): 10000,
        }
        k_directory = {
            (0,3): 1,
            (0,2): 1,
            (0,1): 9,
            (1,2): 4,
            (-1,2): 4,
            (1,1): 16,
            (-1,1): 16,
            (0,0): 10000,
        }
        sum_p = 0
        n = 0
        for cg_inst in self.jindex_sub_pile[0]:
            if cg_inst["e"]:
                sum_p += cg_inst["p"]
                n += 1
        return k_directory[(sum_p, n)]

    def sum_sub_piles_cg_k(self):
        total = np.array([(0,0,0,True)],dtype=cg)
        reg_mul = np.array([(0,0,0,True)],dtype=cg)
        for i in range(len(self.jindex_pile)):
            self.create_sub_pile(i)
            reg_mul[0] = mul_cg_mass(self.get_sub_pile_cg(), self.get_sub_pile_base_k_coeff())
            total[0] = add_cgs(total, reg_mul[0])
        return total[0]

    def stability_index(self):
        #return cg_to_r(self.sum_cg_k())
        return cg_to_r(self.sum_sub_piles_cg_k())
    
    def remove_blocks_by_jindex_list(self, jindex_list):
        for jdex in jindex_list:
            if jdex["e"]:
                self.jindex_pile[jdex["s"],jdex["p"]+self.mid_row]["e"] = False

pile = Pile(g_pile_height, g_blocks_in_row, g_l, g_w, g_h, g_mass)

#print(pile.jindex_pile)
print(pile.stability_index())
#print(pile.create_sub_pile(6))
print("\n")
print("\n")
print("\n")

remove_list = np.array([(0, -1, True), (0, 0, True)], dtype=jindex)

pile.remove_blocks_by_jindex_list(remove_list)
#print(pile.create_sub_pile(6))

print(pile.jindex_pile)
print(pile.stability_index())





################################# EXECUTION






















################################# TRASH

"""
   
    def sum_cg_mass(self, start_s):
        total = np.array([(0,0,0,True)],dtype=cg)
        reg_mul = np.array([(0,0,0,True)],dtype=cg)
        for i in range(start_s, self.pile_height):
            for j in range(self.blocks_in_row):
                reg_mul[0] = mul_cg_mass(self.cg_pile[i,j], self.mass)
                total[0] = add_cgs(total[0], reg_mul[0])
        return total[0]
    
    def sum_mass(self, start_s):
        total = 0
        for i in range(start_s, self.pile_height):
            for j in range(self.blocks_in_row):
                if self.cg_pile[i,j]["e"]:
                    total += self.mass
        return total

    def get_total_cg(self, start_s):
        res_sum_cg_mass = np.array([self.sum_cg_mass(start_s)],dtype=cg)
        res = np.array([(0,0,0,True)],dtype=cg)
        res[0] = mul_cg_mass(res_sum_cg_mass[0], 1/self.sum_mass(start_s))
        return res[0]
    
    def get_k_coeff(self, s):
        k_directory = {
            (0,3): 1,
            (0,2): 1,
            (0,1): 3,
            (1,2): 2,
            (-1,2): 2,
            (1,1): 4,
            (-1,1): 4,
            (0,0): 10000,
        }
        sum_p = 0
        n = 0
        for cg_inst in self.jindex_pile[s]:
            if cg_inst["e"]:
                sum_p += cg_inst["p"]
                n += 1
        return k_directory[(sum_p, n)]

    def sum_cg_k(self):
        total = np.array([(0,0,0,True)],dtype=cg)
        reg_mul = np.array([(0,0,0,True)],dtype=cg)
        for i in range(len(self.jindex_pile)):
            reg_mul[0] = mul_cg_mass(self.get_total_cg(i), self.get_k_coeff(i))
            total[0] = add_cgs(total, reg_mul[0])
        return total[0]
"""