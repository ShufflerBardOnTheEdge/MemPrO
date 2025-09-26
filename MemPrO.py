#Version 6 of MemBrain (WIP)

import os
import warnings
import jax.numpy as jnp
from jax import config
import jax
import numpy as np
import matplotlib.pyplot as plt
import time
from enum import Enum
from functools import partial
import datetime
from jax import tree_util
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
import shutil
from collections import defaultdict


PATH_TO_INSANE = os.environ["PATH_TO_INSANE"]

warnings.filterwarnings('ignore')

#We define some enumerations for use later
"""
class Reses(Enum):
	ALA = 0
	GLY = 1
	ILE = 2
	LEU = 3
	PRO = 4
	VAL = 5
	PHE = 6
	TYR = 7
	TRP = 8
	ARG = 9
	LYS = 10
	HIS = 11
	SER = 12
	THR = 13
	ASP = 14
	GLU = 15
	ASN = 16
	GLN = 17
	CYS = 18
	MET = 19
	UNK = 20
"""
Reses = {"ALA":0,"GLY":1,"ILE":2,"LEU":3,"PRO":4,"VAL":5,"PHE":6,"TYR":7,"TRP":8,"ARG":9,"LYS":10,"HIS":11,"SER":12,"THR":13,"ASP":14,"GLU":15,"ASN":16,"GLN":17,"CYS":18,"MET":19,"UNK":20}
"""
class Beads(Enum):
	BB =  0
	SC1 = 1
	SC2 = 2
	SC3 = 3
	SC4 = 4
	SC5 = 5
	SC6 = 6
"""
Beads = {"BB":0,"SC1":1,"SC2":2,"SC3":3,"SC4":4,"SC5":5,"SC6":6}
"""
class Beadtype(Enum):
	SP2 = 0 #p4
	TC3 = 1 #p4
	SP1 = 2 #P1
	P2 = 3 #p5
	SC2 = 4 #AC2
	SP2a = 5 #p5
	SC3 = 6 #c3
	SC4 = 7 #SC5
	TC5 = 8 #SC5
	TC4 = 9 #SC4
	TN6 = 10 #SP1
	TN6d = 11 #SNd
	SQ3p = 12 #Qp
	SQ4p = 13 #Qp
	TN5a = 14 #SP1
	TP1 = 15 #P1
	SQ5n = 16 #Qa
	Q5n = 17 #Qa
	TC6 = 18 #c5
	C6 = 19 #c5
	P5 = 20 #p5
	SP5 = 21 #p4
	GHOST = 22 #May need for PG as well
"""
Beadtype = {"SP2":0, #p4
	"TC3" : 1, #p4
	"SP1" : 2, #P1
	"P2" : 3, #p5
	"SC2" :4, #AC2
	"SP2a" : 5, #p5
	"SC3" :6, #c3
	"SC4": 7, #SC5
	"TC5" : 8, #SC5
	"TC4" : 9, #SC4
	"TN6" : 10, #SP1
	"TN6d" : 11, #SNd
	"SQ3p" : 12, #Qp
	"SQ4p" : 13, #Qp
	"TN5a" : 14, #SP1
	"TP1" :15, #P1
	"SQ5n" : 16, #Qa
	"Q5n" : 17, #Qa
	"TC6" : 18, #c5
	"C6" : 19, #c5
	"P5" : 20, #p5
	"SP5" : 21, #p4
	"GHOST" : 22} #May need for PG as well}

ResToBead = [["SP2","TC3"],["SP1"],["P2","SC2"],["P2","SC2"],["SP2a","SC3"],["SP2","SC3"],["P2","SC4","TC5","TC5"],["P2","TC4","TC5","TC5","TN6"],["P2","TC4","TN6d","TC5","TC5","TC5"],["P2","SC3","SQ3p"],["P2","SC3","SQ4p"],["P2","TC4","TN6d","TN5a"],["P2","TP1"],["P2","SP1"],["P2","SQ5n"],["P2","Q5n"],["P2","P5"],["P2","SP5"],["P2","TC6"],["P2","C6"],["GHOST"]]
AtomsToBead = [[["N","C","O"],["CB"]],[["N","C","O","CA"]],[["N","C","O"],["CB","CD1","CG1","CG2"]],[["N","C","O"],["CB","CG","CD1","CD2"]],
				 [["C","O"],["CB","CA","CD","CG","N"]],[["N","C","O"],["CB","CG1","CG2"]],[["N","C","O"],["CB","CA","CG"],["CD1","CE1"],["CD2","CE2"]],
				 [["N","C","O"],["CB"],["CD1"],["CD2"],["CZ"]],[["N","C","O"],["CB"],["CG","NE1","CD1"],["CD2","CE2"],["CZ3","CE3"],["CZ2","CH2"]],
				 [["N","C","O"],["CB","CG","CD"],["NE","CZ","NH2","NH1"]],[["N","C","O"],["CB","CG"],["CD","CE","NZ"]],[["N","C","O"],["CB","CG"],["CD2","NE2"],["ND1","CE1"]],
				 [["N","C","O"],["CB","OG"]],[["N","C","O"],["CB","CG2","OG1"]],[["N","C","O"],["CB","CG","OD2"]],[["N","C","O"],["CD","CG","OE2","OE1"]],
				 [["N","C","O"],["CB","CG","ND2"]],[["N","C","O"],["CD","CG","NE2"]],[["N","C","O"],["SG","CB"]],[["N","C","O"],["SD","CE","CG"]],[]]	
		
	
#We need to force JAX to fully utilize a multi-core cpu
no_cpu = int(os.environ["NUM_CPU"])
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count="+str(no_cpu)


#This is useful for debugging nans in grad
#config.update("jax_debug_nans",True)

#forcing JAX to use x64 floats rather than x32
config.update("jax_enable_x64",True)
config.update("jax_platform_name","cpu")

sharding = PositionalSharding(mesh_utils.create_device_mesh((no_cpu,1)))
sharding0 = PositionalSharding(mesh_utils.create_device_mesh((no_cpu,)))
devices = jax.devices()

#This is a class that deals with the pdb files and all construction of position arrays
class PDB_helper:
	def __init__(self,fn,use_weights,build_no,ranges,lsurf,def_surf):
		#A list to help with getting the correct bead types from a cg pdb
		self.beads = ResToBead
		self.beadtype_index = 0
		self.fn = fn
		self.use_weights = use_weights
		self.build_no = build_no
		self.ranges = ranges
		self.lsurf = lsurf
		self.def_surf = def_surf
		
		#Placeholder vars
		self.surface = 0
		self.spheres = 0
		self.surface_poses = 0
		self.bead_types = 0
		self.surf_b_vals = 0
		self.poses = 0
		self.all_poses = 0
		self.b_vals = 0
		self.map_to_beads = 0
		self.normals = 0
		self.all_bead_types = 10
		
		#We need the following two methods to tell jax what is and what is not static
	def _tree_flatten(self):
		children = (self.surface,self.spheres,self.surface_poses,self.bead_types,self.surf_b_vals,self.poses,self.all_poses,self.b_vals,self.map_to_beads,self.normals,self.all_bead_types)
		aux_data = {"Beads":self.beads,"fn":self.fn,"weights":self.use_weights,"Build_no":self.build_no,"ranges":self.ranges,"Lsurf":self.lsurf,"Dsurf":self.def_surf}
		return (children,aux_data)
	
	@classmethod
	def _tree_unflatten(cls,aux_data,children):
		obj = cls(aux_data["fn"],aux_data["weights"],aux_data["Build_no"],aux_data["ranges"],aux_data["Lsurf"],aux_data["Dsurf"])
		obj.surface = children[0]
		obj.spheres = children[1]
		obj.surface_poses = children[2]
		obj.bead_types = children[3]
		obj.surf_b_vals = children[4]
		obj.poses = children[5]
		obj.all_poses = children[6]
		obj.b_vals = children[7]
		obj.map_to_beads = children[8]
		obj.normals = children[9]
		obj.all_bead_types = children[10]
		return obj

		
		
	def in_ranges(self,num):
		for i in self.ranges:
			if(i[1] > num >= i[0]):
				return True
		return False
	#detects format of input
	def detect_atomistic(self):
		lfile = open(os.path.join(self.fn),"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		for c in content:
			if(len(c) > 46):
				if("[" not in c and c[:4]=="ATOM"):	
					bead = c[12:17].strip()
					if(bead in ["BB","SC1","SC2","SC3","SC4","SC5"]):
						return False
		return True
	
	#converts a atomistic resiude to cg representation
	#This is by no means a good coarse graining but as there is no MD involved and only approximate locations are needed
	#this is good enough. Orientations using this CG rep and a Martinized CG rep are very similar.
	def convert_to_cg(self,res_type,poses,b_vals,atom_types):
		if(res_type == 20):
			num = 1
			atom_to_bead = atom_types
		else:
			atom_to_bead = AtomsToBead[res_type]
			num = len(atom_to_bead)
		
		reses = np.zeros(num)+res_type
		beads_pos = np.zeros((num,3))
		avs = np.zeros(num)
		bead_types = np.zeros(num)
		new_bvals = np.zeros(num)
		map_to_beads = np.zeros(len(poses))
		for i in range(num):
			bead_types[i] = Beadtype[self.beads[res_type][i]]
		for i,xi in enumerate(poses):
			#print("here",atom_types[i])
			for k in range(num):
				if(atom_types[i] in atom_to_bead[k]):
					beads_pos[k] += xi
					avs[k] += 1
					new_bvals[k] += b_vals[i]
					map_to_beads[i] = k
		bad = []
		good = []
		for i in range(num):
			if(avs[i] != 0):
				beads_pos[i] /= avs[i]
				new_bvals[i] /= avs[i]
				good.append(i)
			else:
				bad.append(i)
		if(len(good)==0):
			pass
		for i in bad:
			beads_pos[i] = np.mean(beads_pos[good],axis=0)
			new_bvals[i] = np.mean(new_bvals[good])
		return np.array(beads_pos),np.array(bead_types),np.array(reses),np.array(new_bvals),map_to_beads


	#Loads a atomistic pdb into a cg representation
	def load_atomistic_pdb(self):
		self.reses = []
		self.poses = np.empty((0,3))
		self.bead_types = np.empty((0))
		self.b_vals = []
		self.map_to_beads = np.empty(0)
		reses2 = []
		poses2 = []
		self.all_poses = []
		atom_types = []
		b_vals2 = []
		lfile = open(os.path.join(self.fn),"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		prev_atom_num = -1
		for c in content:
			if(len(c) > 46):
				if("[" not in c and c[:4]=="ATOM"):	
					res = c[17:21].strip()
					bead = c[12:17].strip()
					zpos = c[46:54]
					ypos = c[38:46]
					xpos = c[30:38]
					b_val = float(c[60:66].strip())
					atom_num = int(c[22:26].strip())
					pos = np.array([float(xpos.strip()),float(ypos.strip()),float(zpos.strip())])
					if(not np.any(np.isnan(pos))):
						if(res not in Reses.keys()):
							res = "UNK" 
						self.all_poses.append(pos)
						if(atom_num == prev_atom_num):
							atom_types.append(bead)
							reses2.append(Reses[res])
							poses2.append(pos)
							if(self.ranges.size != 0):
								if(self.in_ranges(atom_num)):
									b_vals2.append(1.0)
								else:
									b_vals2.append(0)
							else:
								b_vals2.append(b_val)
						elif(prev_atom_num != -1):
							bead_pos,bead_types2,reses3,b_vals2,m2b = self.convert_to_cg(reses2[0],poses2,b_vals2,atom_types)
							self.map_to_beads = np.concatenate((self.map_to_beads,m2b+self.bead_types.shape[0]))	
							self.poses =np.concatenate((self.poses,bead_pos))
							self.bead_types = np.concatenate((self.bead_types,bead_types2))
							self.reses = np.concatenate((self.reses,reses3))
							self.b_vals = np.concatenate((self.b_vals,b_vals2))		
							reses2 = []
							poses2 = []
							b_vals2 = []
							atom_types = []
							atom_types.append(bead)
							reses2.append(Reses[res])
							poses2.append(pos)
							if(self.ranges.size != 0):
								if(self.in_ranges(atom_num)):
									b_vals2.append(1.0)
								else:
									b_vals2.append(0)
							else:
								b_vals2.append(b_val)
						else:
							atom_types.append(bead)
							reses2.append(Reses[res])
							poses2.append(pos)
							if(self.ranges.size != 0):
								if(self.in_ranges(atom_num)):
									b_vals2.append(1.0)
								else:
									b_vals2.append(0)
							else:
								b_vals2.append(b_val)
						prev_atom_num = atom_num
		 
		bead_pos,bead_types2,reses3,b_vals2,m2b = self.convert_to_cg(reses2[0],poses2,b_vals2,atom_types)
		self.map_to_beads = np.concatenate((self.map_to_beads,m2b+self.bead_types.shape[0]))	
		self.poses =np.concatenate((self.poses,bead_pos))
		self.bead_types = np.concatenate((self.bead_types,bead_types2))
		self.reses = np.concatenate((self.reses,reses3))
		self.b_vals = np.concatenate((self.b_vals,b_vals2))	
		self.b_vals = jnp.array(self.b_vals)	
		self.poses = jnp.array(self.poses)
		self.all_poses = jnp.array(self.all_poses)
		self.reses = jnp.array(self.reses)
		self.bead_types = jnp.array(self.bead_types)
		pos_mean = np.mean(self.poses,axis=0)
		self.poses = self.poses - pos_mean
		self.all_poses = self.all_poses - pos_mean
		self.b_vals = self.b_vals/jnp.max(self.b_vals+1e-9)
		return pos_mean		
		

	def load_cg_pdb(self):
		self.reses = []
		self.poses = []
		self.bead_types = []
		self.b_vals = []

		# Read PDB content
		with open(os.path.join(self.fn), "r") as lfile:
			content = lfile.read().split("\n")

		# Group atom lines by (residue name, residue ID)
		residue_atoms = defaultdict(list)
		for c in content:
			if len(c) > 54 and "[" not in c and c.startswith("ATOM"):
				resname = c[17:21].strip()
				resid = int(c[22:26].strip())
				chain_id = c[21].strip()
				residue_atoms[(resname, chain_id, resid)].append(c)

		for (resname, chain_id, resid), atoms in residue_atoms.items():
			original_resname = resname
			if resname not in Reses:
				print(f"Unknown residue {resname}, using UNK")
				resname = "UNK"
			res_index = Reses[resname]
			bead_list = ResToBead[res_index]

			# Collect atom positions
			atom_pos_list = []
			atom_bval_list = []
			atom_nums = []

			for c in atoms:
				try:
					x = float(c[30:38].strip())
					y = float(c[38:46].strip())
					z = float(c[46:54].strip())
					b = float(c[60:66].strip())
					atom_num = int(c[22:26].strip())
					pos = np.array([x, y, z])

					if not np.any(np.isnan(pos)):
						atom_pos_list.append(pos)
						atom_bval_list.append(b)
						atom_nums.append(atom_num)
				except ValueError:
					continue

			if len(atom_pos_list) == 0:
				print(f"Skipping residue {resname} {resid} due to no valid positions.")
				continue

			# Fallback mean values
			mean_pos = np.mean(atom_pos_list, axis=0)
			mean_b = np.mean(atom_bval_list) if atom_bval_list else 0.0

			# Use quick hack: match bead types with atom positions by order
			if len(bead_list) != len(atom_pos_list):
				print(atom_pos_list)
				print(f"Mismatch in bead count vs atom count for {resname} {resid}. Using mean positions. Expecting {len(bead_list)}, found {len(atom_pos_list)}")
				for bead_type in bead_list:
					bead_index = Beadtype.get(bead_type, -1)
					if bead_index == -1:
						raise ValueError(f"Unknown bead type: {bead_type}")
					self.bead_types.append(bead_index)
					self.reses.append(res_index)
					self.poses.append(mean_pos)
					self.b_vals.append(mean_b)
				continue

			for bead_type, pos, b_val in zip(bead_list, atom_pos_list, atom_bval_list):
				bead_index = Beadtype.get(bead_type, -1)
				if bead_index == -1:
					raise ValueError(f"Unknown bead type: {bead_type}")
				self.bead_types.append(bead_index)
				self.reses.append(res_index)
				self.poses.append(pos)

				if self.ranges.size != 0:
					in_range = any(self.in_ranges(num) for num in atom_nums)
					self.b_vals.append(1.0 if in_range else 0.0)
				else:
					self.b_vals.append(b_val)

		# Convert to JAX arrays
		self.poses = jnp.array(self.poses)
		pos_mean = jnp.mean(self.poses, axis=0)
		self.poses -= pos_mean

		self.reses = jnp.array(self.reses)
		self.bead_types = jnp.array(self.bead_types)
		self.b_vals = jnp.array(self.b_vals)
		self.b_vals /= jnp.max(self.b_vals + 1e-9)

		self.all_poses = self.poses.copy()
		self.map_to_beads = jnp.arange(self.poses.shape[0])

		return pos_mean



	#loads a pdb
	def load_pdb(self):
		if(self.detect_atomistic()):
			print("Atomistic file detected")
			if(self.build_no > 0):
				print("WARNING: Cannot build a CG-System for MD with atomistic input")
			self.build_no = 0
			pos_mean = self.load_atomistic_pdb()
		else:
			print("Coarse Grained file detected")
			pos_mean = self.load_cg_pdb()
		#Setting read b-vals to 1 if not using weights
		if(not self.use_weights and self.ranges.size == 0):
			self.b_vals = self.b_vals.at[:].set(1.0)
		return pos_mean

	### Begining of code for getting the surface of the CG protein ###

	#We create a sphere using a fibonacci spiral lattice to ensure a even distribution 
	def create_ball(self,brad,bsize):
		self.ball = jnp.zeros((bsize,3))
		gr = (1+jnp.sqrt(5))/2
		def ball_fun_3(ball,ind):
			phi = jnp.arccos(1-2*(ind+0.5)/(bsize))
			theta = jnp.pi*(ind+0.5)*(gr)*2
			ball = ball.at[ind].set(jnp.array([brad*jnp.cos(phi),brad*jnp.sin(phi)*jnp.sin(theta),brad*jnp.sin(phi)*jnp.cos(theta)]))
			return ball, ind
			
		self.ball,_ = jax.lax.scan(ball_fun_3,self.ball,np.arange(bsize))
		return self.ball
		
	@jax.jit
	def fire_pointv3(self,fpoint,direc,points,hit_points):
		rad = 7
		best_ind = jnp.array([0.0,1e9])
		direc = direc/jnp.linalg.norm(direc)
		@jax.vmap
		def fp3_fun(ind):
			pdir = points[ind]-fpoint
			dist = jnp.abs(jnp.dot(pdir,direc))
			side_dist = jnp.abs(jnp.power(jnp.linalg.norm(pdir),2)-dist*dist)
			test_val1 = rad*rad-side_dist
			distance = dist-jnp.sqrt(test_val1)
			return distance
			
		shard_nn = jax.lax.with_sharding_constraint(jnp.arange(points.shape[0]),sharding0)
		disters = fp3_fun(shard_nn)
		inder = jnp.nanargmin(disters)
		hit_points = hit_points.at[inder].set(hit_points[inder]+1)
		return hit_points
		
		
		
		
		
	@partial(jax.jit,static_argnums=(1,2))	
	def fire_n_points(self,nn,nps,fpoints,direcs,points,hit_points):
		@jax.vmap
		def fnp(ind):
			ret_points = jax.lax.with_sharding_constraint(jnp.zeros(nps),sharding0)
			ret_points = self.fire_pointv3(fpoints[ind],direcs[ind],points,ret_points)
			return ret_points
			
		shard_nn = jax.lax.with_sharding_constraint(jnp.arange(nn),sharding0)
		ret_points = fnp(shard_nn)
		hit_points = hit_points.at[:,0].set( hit_points[:,0]+ jnp.sum(ret_points,axis=0))
		return hit_points
				
	def get_surface_v2(self,points,hit_points,def_surf):
		nnn = 5000
		nnnr = nnn%no_cpu
		nnn -= nnnr
		no_runs = nnn//no_cpu
		maxer = jnp.max(jnp.linalg.norm(points,axis=1))
	
		mrange = int(maxer*2+20)
		chh1 = nnn
		chh = chh1
		while chh/chh1 > 0.0025:
			if def_surf:
				dpoints = np.random.rand(nnn,2)
				dpoints[:,0] *= 2*np.pi
				dpoints[:,1] = np.pi/2
				#dpoints[:,1] -= np.pi/4
			else:
				dpoints = np.random.rand(nnn,2)*2*np.pi
			dpoints2 = np.array([[np.sin(i[0]),np.cos(i[0])*np.sin(i[1]),np.cos(i[0])*np.cos(i[1])] for i in dpoints])
			inder = np.array([np.random.randint(0,points.shape[0]) for i in range(nnn)],dtype=int)
			fpoints = jnp.array(mrange*dpoints2+points[inder])
			direcs = jnp.array(-dpoints2)
			start_h = hit_points[hit_points==0].shape[0]			
			hit_points = self.fire_n_points(nnn,hit_points.shape[0],fpoints,direcs,points,hit_points)
			chh = start_h - hit_points[hit_points==0].shape[0]

		hit_points = hit_points.at[hit_points>0].set(1)
		return hit_points



	@partial(jax.jit,static_argnums=3)
	def get_surface_v3(self,points,bpoints,mrad):
		srad = 4
		thresh = 0.9
		total = jnp.zeros((40,20))
		total = get_sph_circ(total,jnp.array([0,0]),jnp.pi)
		total_area = jnp.sum(total)
		#jax.debug.print("TA{x}",x=total_area)
		is_surf = jnp.zeros((points.shape[0]))
		def sloopv3_1(is_surf,ind):
			pind = ind
			ang_grid = jnp.zeros((40,20))
			def sloopv3_2(ang_grid,ind):
				direc = points[pind]-points[bpoints[pind,ind]]
				dist = jnp.linalg.norm(direc)
				def far(ang_grid):
					return ang_grid
				def nfar(ang_grid):
					#angle = (jnp.pi/3)*srad/jnp.max(jnp.array([4.0,dist]))#jnp.arccos(2*dist/srad)#0.6
					angle = jnp.arccos(jnp.clip(dist/(2*srad),0,1))
					cen_y = jnp.arctan2(direc[2],jnp.sqrt(direc[0]*direc[0]+direc[1]*direc[1]))
					cen_x = jnp.arctan2(direc[1],direc[0])
					ang_grid = get_sph_circ(ang_grid,jnp.array([cen_x,cen_y]),angle)
					return ang_grid
				ang_grid = jax.lax.cond((dist < srad*3)*(dist>1e-5),nfar,far,ang_grid)
				return ang_grid,ind
			ang_grid,_ = jax.lax.scan(sloopv3_2,ang_grid,jnp.arange(mrad))
			perc = jnp.sum(ang_grid)/total_area
			#jax.debug.print("Perc {x}",x=perc)
			def gthresh(is_surf):
				is_surf = is_surf.at[ind].set(1)
				return is_surf
			def lthresh(is_surf):
				return is_surf
			is_surf = jax.lax.cond(perc<thresh,gthresh,lthresh,is_surf)
			return is_surf,ind
		is_surf,_=jax.lax.scan(sloopv3_1,is_surf,jnp.arange(points.shape[0]))
		return is_surf
		
	#This function gets the surface residues of a given cg protein
	def get_surface(self):
		bsize = 19#19#38
		brad = 4
		self.create_ball(brad,bsize)
		pos_num = self.poses.shape[0]
		new_num = pad_num(self.poses)
		
		
		self.poses = jnp.pad(self.poses,((0,new_num-pos_num),(0,0)),mode="edge")
		
		#Calculating normals at each point of the cg protein
		timer = time.time()
		@partial(jax.vmap)
		def normal_fun_2(ind):
			def normal_fun_1(ind_fix,ind):
				normal_count = jnp.zeros(4)
				def in_sphere(normal,posa,posb):
					normal = normal.at[3].set(normal[3]+1)
					normal = normal.at[:3].set(normal[:3]+(-posa+posb))
					return normal
				def not_in_sphere(normal,posa,posb):
					return normal
				normal = jax.lax.cond(jnp.linalg.norm(self.poses[ind]-self.poses[ind_fix])<32,in_sphere,not_in_sphere,normal_count,self.poses[ind],self.poses[ind_fix])#8####################<--------------------------
				return ind_fix,normal
				
			_,normals = jax.lax.scan(normal_fun_1,ind,jnp.arange(self.poses.shape[0]))
			normals_sum = jnp.sum(normals,axis=0)
			normal = normals_sum[:3]/normals_sum[3]
			normal = normal/jnp.linalg.norm(normal)
			return normal
		shard_nns = jax.device_put(jnp.arange(self.poses.shape[0]),sharding0)
		normals = normal_fun_2(shard_nns)
		
		self.normals = jnp.array(jax.device_get(normals))#jax.device_put(normals)#jax.device_put_sharded([normals],[devices[0]])

		jax.block_until_ready(self.normals)
		self.surface = jnp.zeros((self.poses.shape[0],1))
		
		
		if(self.lsurf):
			bpoints,mrad = get_binned_points(np.array(self.poses),16)
			is_surf = self.get_surface_v3(jnp.array(self.poses),jnp.array(bpoints),mrad).block_until_ready()
			self.surface = self.surface.at[:,0].set(jnp.array(is_surf))
		else:
			shard_points = jax.device_put(self.poses,sharding)
			shard_surface = jax.device_put(self.surface,sharding)
			shard_surface = self.get_surface_v2(shard_points,shard_surface,self.def_surf).block_until_ready()
			self.surface = jnp.array(jax.device_get(shard_surface))
		
		surface_number = jnp.sum(self.surface,dtype = "int")
		#A new array is created containg only surface positions
		self.surface_poses = self.poses[self.surface[:,0] == 1]

		timer = time.time()
		#Each point on the surface had an asocciated sphere indicating directions which are exposed
		#Here we flag the directions which are free
		@partial(jax.vmap)
		def sphere_fun_3(ind):
			def sphere_fun_2(ind_fix,ind):
				def sphere_fun_1(ind_fix,ind):
					def in_prot():
							return 0
					def not_in_prot():
						return 1
					return ind_fix,jax.lax.cond(jnp.linalg.norm(self.poses[ind_fix[0]]+normals[ind_fix[0]]*4-self.poses[ind]+self.ball[ind_fix[1]])<4,in_prot,not_in_prot)
				_,on_surf_arr=jax.lax.scan(sphere_fun_1,jnp.array([ind_fix,ind]),jnp.arange(self.poses.shape[0]))
				on_surf_arr = jnp.array(jax.device_get(on_surf_arr))
				
				on_surf = jnp.prod(on_surf_arr)
				return ind_fix,on_surf
			_,on_surf_ball = jax.lax.scan(sphere_fun_2,ind,jnp.arange(self.ball.shape[0]))
			ball_arr = on_surf_ball*self.surface[ind]
			return ball_arr
		
		
		shard_nn = jax.device_put(jnp.arange(self.poses.shape[0]),sharding0)
		spheres = sphere_fun_3(shard_nn)
		
		self.poses = self.poses[:pos_num]
		self.surface = self.surface[:pos_num,0]
		#Here we set some values associated with each bead to be for surface only
		surface_sphere = spheres[:pos_num][self.surface == 1]
		self.all_bead_types = self.bead_types.copy()
		self.bead_types = self.bead_types[self.surface == 1]
		self.surf_b_vals = self.b_vals[self.surface == 1]
		
		jax.block_until_ready(spheres)
		jax.block_until_ready(self.poses)
		
		#Here we set the cartesean directions for each sphere (as before we only had flags)
		self.spheres = jnp.zeros((surface_number,self.ball.shape[0],3))
		def set_sphere_poses_fun_2(sphere_poses,ind):
			timeser = jnp.transpose(jnp.array([surface_sphere[ind],surface_sphere[ind],surface_sphere[ind]]))
			sphere_poses = sphere_poses.at[ind].set(sphere_poses[ind,:]+timeser*self.ball)
			return sphere_poses,ind
		self.spheres,_ = jax.lax.scan(set_sphere_poses_fun_2,self.spheres,jnp.arange(surface_sphere.shape[0]))
		surf_mean = jnp.mean(self.surface_poses,axis=0)
		self.surface_poses = self.surface_poses-surf_mean
		self.poses = self.poses-surf_mean
		self.all_poses = self.all_poses-surf_mean
		self.bead_types = jnp.array(self.bead_types,dtype="int")
		
		
	#For perisplasm spanning proteins it starts oriented with its farthest point at [0,0,1]
	def starting_orientation(self):
		all_dists = jnp.linalg.norm(self.surface_poses,axis=1)
		max_ind = jnp.argmax(all_dists)
		far = self.surface_poses[max_ind]
		far_dir = far/jnp.linalg.norm(far)
		ang2 = jnp.arccos(jnp.dot(far_dir,jnp.array([0,0,1])))
		far_projxy = far_dir.at[2].set(0)
		far_projxy /= jnp.linalg.norm(far_projxy)
		ang1 = jnp.arccos(jnp.dot(far_projxy,jnp.array([0,1,0])))
		self.surface_poses = position_point_jit(0,-ang1,ang2,self.surface_poses)
		self.poses = position_point_jit(0,-ang1,ang2,self.poses)
		self.all_poses = position_point_jit(0,-ang1,ang2,self.all_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_point_jit(0,-ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,self.spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
	
		
	def starting_orientation_v2(self):
		all_dists = jnp.linalg.norm(self.surface_poses,axis=1)
		max_ind = jnp.argmax(all_dists)
		max_inds = jnp.argsort(all_dists)
		far = self.surface_poses[max_ind]
		
		all_dists2 = jnp.linalg.norm(self.surface_poses-far,axis=1)
		min_inds = jnp.argsort(all_dists2)
		
		allowed1 = min_inds[all_dists2[min_inds]<jnp.linalg.norm(far)]
		
		allowed2 = max_inds[all_dists[max_inds]>jnp.linalg.norm(far)*0.85]
		all_allowed = jnp.intersect1d(allowed1,allowed2)
		
		
		far = jnp.mean(self.surface_poses[all_allowed],axis=0)
		far_dir = far/jnp.linalg.norm(far)
		ang2 = jnp.arccos(jnp.dot(far_dir,jnp.array([0,0,1])))
		far_projxy = far_dir.at[2].set(0)
		far_projxy /= jnp.linalg.norm(far_projxy)
		ang1 = jnp.arccos(jnp.dot(far_projxy,jnp.array([0,1,0])))
		
		self.surface_poses = position_point_jit(0,-ang1,ang2,self.surface_poses).block_until_ready()
	
		self.poses = position_point_jit(0,-ang1,ang2,self.poses).block_until_ready()
	
		self.all_poses = position_point_jit(0,-ang1,ang2,self.all_poses).block_until_ready()
		
		def rot_spheres(carry,ind):
			new_spheres = position_point_jit(0,-ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,self.spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		
		return 0
	
	#Use for testing different input orientations
	def test_start(self,new_pos):
		in_depth,ang1,ang2 = new_pos
		self.surface_poses = position_point_jit(in_depth,ang1,ang2,self.surface_poses)
		self.poses = position_point_jit(in_depth,ang1,ang2,self.poses)
		self.all_poses = position_point_jit(in_depth,ang1,ang2,self.all_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_point_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,self.spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		
	def recenter_bvals(self):
		tot = jnp.zeros(4) 
		def recenter_fun(tot,ind):
			def bzero(tot):
				return tot
			def not_bzero(tot):
				tot = tot.at[:3].set(tot[:3]+self.surface_poses[ind])
				tot = tot.at[3].set(tot[3]+1)
				return tot
			tot = jax.lax.cond(self.surf_b_vals[ind]<1e-5,bzero,not_bzero,tot)
			return tot,ind
		tot,_ = jax.lax.scan(recenter_fun,tot,jnp.arange(self.surface_poses.shape[0]))
		surf_mean = tot[:3]/tot[3]
		self.surface_poses = self.surface_poses-surf_mean
		self.poses = self.poses-surf_mean
		self.all_poses = self.all_poses-surf_mean
	
	def get_data(self):		
		return (self.surface,self.spheres,self.surface_poses,self.bead_types,self.surf_b_vals,self.poses,self.all_poses,self.b_vals,self.build_no,np.array(self.map_to_beads,dtype=int),self.normals,self.all_bead_types)
		
#This is the main orientation class
class MemBrain:
	def __init__(self,data,int_data,peri,force_calc,pg_layer_pred,mem_data,gcurv,dbmem,write_bfac,rank_sort):
		#We use interactions strengths from martini using a POPE(Q4p)/POPG(P4)/POPC(Q1)? lipid as a template
		self.int_data = int_data
		self.W_B_mins = int_data[0]
		self.LH1_B_mins = int_data[1]
		self.LH2_B_mins = int_data[2]
		self.LH3_B_mins = int_data[3]
		self.LH4_B_mins = int_data[4]
		self.LT1_B_mins = int_data[5]
		self.LT2_B_mins = int_data[6]
		self.Charge_B_mins =int_data[7]
		
		
		self.mem_data = mem_data
		
		self.charge_mult = mem_data[0]
		self.charge_mult_om = mem_data[1]

		#These values are hyper parameters for the minimisation algorithm. The values here were found mostly by trial and error
		#It is not reccomended to change these.
		self.gamma_val = 0.999
		self.lr_pos = 4e-5
		self.lr_heights = 0.025
		self.lr_cens = 0.02
	
		#We need to define the PG layer differently (as it is outdated)
		self.pg_layer = -jnp.array([4.61,4.61,4.64,5.0,2.62,5.0,2.66,2.89,2.89,2.71,4.64,4.32,4.88,4.88,4.64,4.64,4.71,4.71,3.28,3.28,5.0,4.61,0.0])
		self.pg_charge = 0.008
		self.pg_water = -jnp.array([5.0,5.0,4.5,5.6,2.3,5.6,2.7,3.1,3.1,2.7,4.5,4.5,5.6,5.6,4.5,4.5,5.6,5.6,3.1,3.1,5.6,5.0,0.0])
		self.pg_thickness = 40
		self.sheet_charge = 0.000016
		
		#We now define the structure of the membrane (Possibly can be defined in an input file in future)
		self.memt_tails = mem_data[2]
		self.memt_heads = 10.0
		self.memt_total = self.memt_tails+self.memt_heads
		
		h1_w = self.memt_heads/6.0
		h2_w = self.memt_heads/6.0
		h3_w = self.memt_heads/6.0
		
		l_w = self.memt_tails
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		self.mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		self.mem_structure = self.mem_structure_im.copy()
		
		self.memt_tails_om = mem_data[3]
		self.memt_heads_om = 10.0
		self.memt_total_om = self.memt_tails_om+self.memt_heads_om
		
		h1_w_om = self.memt_heads_om/6.0
		h2_w_om = self.memt_heads_om/6.0
		h3_w_om = self.memt_heads_om/6.0
		
		l_w_om = self.memt_tails_om
		meml_om = -l_w_om/2.0 -h1_w_om -h2_w_om-h3_w_om
		self.mem_structure_om = jnp.array([meml_om,meml_om+h1_w_om,meml_om+h2_w_om+h1_w_om,meml_om+h2_w_om+h1_w_om+h3_w_om,meml_om+h2_w_om+h1_w_om+h3_w_om+l_w_om,meml_om+h2_w_om+h1_w_om+2*h3_w_om+l_w_om,meml_om+2*h2_w_om+h1_w_om+2*h3_w_om+l_w_om,meml_om+2*h2_w_om+2*h1_w_om+2*h3_w_om+l_w_om])
		self.mem_struture_dm = [self.mem_structure_im,self.mem_structure_om]
		
		
		#Here we set all the data produced by PDB_Helper
	
		self.data = data
		pad_num(data[0])
		self.surface = data[0]
		self.spheres = data[1]
		self.surface_poses = data[2]
		self.bead_types = data[3]
		self.surf_b_vals = data[4]
		self.poses = data[5]
		self.all_poses = data[6]
		self.b_vals = data[7]
		self.map_to_beads = data[9]
		
		
		#We need a flag for peripheral proteins
		self.peri = peri
		
		#A flag for preforming force calculationss
		self.force_calc  = force_calc
		
		#We need to flag if a curvature minimisation has occured
		self.curva = False
		
		#A flag for predicting the PG cell wall
		self.pg_layer_pred = pg_layer_pred
		
		#Flag for membrane thickness optimisation
		self.gcurv = gcurv
		
		#flag for dm
		self.dbmem = dbmem
		
		#flag to write pots to bfacs
		self.write_bfac = write_bfac
		
		self.rank_sort = rank_sort
		
		self.build_no = data[8]
		
		#We define some fixed values for curvature minimisation
		self.numa = 25
		self.numb = 25
		self.cut_a = 12
		self.cut_b = 12
		self.nums = 5
		self.lss_a = jnp.array([0.03]*self.numa)
		self.lss_b = jnp.array([0.03]*self.numb)#0.03
		
		
		#We define some values for limiting curvature mins (Memory/Speed reasons)
		self.only_top = 10000
		self.red_cpus = no_cpu

		
		#Placeholder vars TODO remove vars ony longer in use
		self.no_mins=0
		self.result_grid = jnp.zeros((1,1))
		self.start_grid = jnp.zeros((1,1))
		self.minima = jnp.zeros((1,8))
		self.pg_poses=0
		self.all_areas=0
		self.re_rank_vals = 0
		self.re_rank_pots = 0
		self.re_rank_disses = 0
	
	#We need the following two methods to tell jax what is and what is not static
	def _tree_flatten(self):
		children = (self.mem_structure,self.result_grid,self.curva,self.mem_structure_im,self.mem_structure_om,self.minima,self.pg_poses,self.minima,self.all_areas,self.re_rank_vals,self.re_rank_pots,self.re_rank_disses,self.start_grid)
		aux_data = {"int_data":self.int_data,"Data":self.data,"Poses":self.poses,"Bead_types":self.bead_types,"All_Poses":self.all_poses,
		"B_vals":self.b_vals,"Surface_poses":self.surface_poses,"Sphere_poses":self.spheres,
			"Surface":self.surface,"Surface_b_vals":self.surf_b_vals,
			"WBmins":self.W_B_mins,"LH1B_mins":self.LH1_B_mins,"LH2B_mins":self.LH2_B_mins,
			"LH3B_mins":self.LH3_B_mins,"LH4B_mins":self.LH4_B_mins,"LT1B_mins":self.LT1_B_mins,
			"LT2B_mins":self.LT2_B_mins,"ChargeBmins":self.Charge_B_mins,"Peri":self.peri,"numa":self.numa,
			"numb":self.numb,"cuta":self.cut_a,"cutb":self.cut_b,"Nums":self.nums,"top_min":self.only_top,
			"red_cpus":self.red_cpus,"lssa":self.lss_a,"lssb":self.lss_b,"no_mins":self.no_mins,"force_calc":self.force_calc,"pg_layer":self.pg_layer_pred,
			"mem_Data":self.mem_data,"memop":self.gcurv,"cm":self.charge_mult,"cmo":self.charge_mult_om,"dbmem":self.dbmem,"write_b":self.write_bfac,"rsort":self.rank_sort}
		return (children,aux_data)
	
	@classmethod
	def _tree_unflatten(cls,aux_data,children):
		obj = cls(aux_data["Data"],aux_data["int_data"],aux_data["Peri"],aux_data["force_calc"],aux_data["pg_layer"],aux_data["mem_Data"],aux_data["memop"],aux_data["dbmem"],aux_data["write_b"],aux_data["rsort"])
		obj.mem_structure = children[0]
		obj.mem_structure_im = children[3]
		obj.mem_structure_om = children[4]
		obj.no_mins = aux_data["no_mins"]
		obj.result_grid = children[1]
		obj.numa = aux_data["numa"]
		obj.numb = aux_data["numb"]
		obj.cut_a = aux_data["cuta"]
		obj.cut_b = aux_data["cutb"]
		obj.curva = children[2]
		obj.lss_a = aux_data["lssa"]
		obj.lss_b = aux_data["lssb"]
		obj.charge_mult = aux_data["cm"]
		obj.charge_mult_om = aux_data["cmo"]
		obj.minima = children[5]
		obj.pg_poses = children[6]
		obj.all_areas = children[7]
		obj.re_rank_vals = children[8]
		obj.re_rank_pots = children[9]
		obj.re_rank_disses = children[10]
		obj.start_grid = children[11]
		return obj



	### Begining of orientation code ###
	
	#We use sigmoids to create a smoothly varying funtion to determin the potential of a single point
	@jax.jit
	def smjh(self,x,grad,bead_num,mem_structure):
		bd1 = (1.0-sj(x-mem_structure[0],grad))*self.W_B_mins[bead_num]+sj(x-mem_structure[0],grad)*self.LH1_B_mins[bead_num]
		bd2 = (1.0-sj(x-mem_structure[1],grad))*bd1+sj(x-mem_structure[1],grad)*self.LH2_B_mins[bead_num]
		bd3 = (1.0-sj(x-mem_structure[2],grad))*bd2+sj(x-mem_structure[2],grad)*self.LH3_B_mins[bead_num]
		bd4 = (1.0-sj(x-mem_structure[3],grad))*bd3+sj(x-mem_structure[3],grad)*self.LT1_B_mins[bead_num]
		bd5 = (1.0-sj(x-mem_structure[4],grad))*bd4+sj(x-mem_structure[4],grad)*self.LH3_B_mins[bead_num]
		bd6 = (1.0-sj(x-mem_structure[5],grad))*bd5+sj(x-mem_structure[5],grad)*self.LH2_B_mins[bead_num]
		bd7 = (1.0-sj(x-mem_structure[6],grad))*bd6+sj(x-mem_structure[6],grad)*self.LH1_B_mins[bead_num]
		bd8 = (1.0-sj(x-mem_structure[7],grad))*bd7+sj(x-mem_structure[7],grad)*self.W_B_mins[bead_num]
		return bd8-self.W_B_mins[bead_num]
		
	#There are 2 options here. One of them considers the hydration of the lipid head region the other does not. It does not seem
	#to have much of an impact which is used, likley as the main driving force comes from the hydrophobicity.
	@jax.jit
	def smj(self,x,grad,bead_num,mem_structure):
		l1min = (self.W_B_mins[bead_num]+self.LH1_B_mins[bead_num]-jnp.abs(self.W_B_mins[bead_num]-self.LH1_B_mins[bead_num]))/2
		l2min = (self.W_B_mins[bead_num]+self.LH2_B_mins[bead_num]-jnp.abs(self.W_B_mins[bead_num]-self.LH2_B_mins[bead_num]))/2
		l3min = (self.W_B_mins[bead_num]+self.LH3_B_mins[bead_num]-jnp.abs(self.W_B_mins[bead_num]-self.LH3_B_mins[bead_num]))/2
		bd1 = (1.0-sj(x-mem_structure[0],grad))*self.W_B_mins[bead_num]+sj(x-mem_structure[0],grad)*l1min
		bd2 = (1.0-sj(x-mem_structure[1],grad))*bd1+sj(x-mem_structure[1],grad)*l2min
		bd3 = (1.0-sj(x-mem_structure[2],grad))*bd2+sj(x-mem_structure[2],grad)*l3min
		bd4 = (1.0-sj(x-mem_structure[3],grad))*bd3+sj(x-mem_structure[3],grad)*self.LT1_B_mins[bead_num]
		bd5 = (1.0-sj(x-mem_structure[4],grad))*bd4+sj(x-mem_structure[4],grad)*l3min
		bd6 = (1.0-sj(x-mem_structure[5],grad))*bd5+sj(x-mem_structure[5],grad)*l2min
		bd7 = (1.0-sj(x-mem_structure[6],grad))*bd6+sj(x-mem_structure[6],grad)*l1min
		bd8 = (1.0-sj(x-mem_structure[7],grad))*bd7+sj(x-mem_structure[7],grad)*self.W_B_mins[bead_num]
		return bd8-self.W_B_mins[bead_num]
		
	
	#This is a function for calculating an estimate of the potential of the PG layer at a point
	@jax.jit
	def pg_pot(self,x,grad,bead_num):
		bd1 = (1.0-sj(x+self.pg_thickness/2,grad))*self.pg_water[bead_num]+sj(x+self.pg_thickness/2,grad)*self.pg_layer[bead_num]
		bd2 = (1.0-sj(x-self.pg_thickness/2,grad))*bd1+sj(x-self.pg_thickness/2,grad)*self.pg_water[bead_num]
		return bd2-self.pg_water[bead_num]

		
	#We define new functions that let us have a double membrane (OM/IM)
	@jax.jit
	def dmsj(self,x,zdist,grad,bead_num,mem_structure_im,mem_structure_om):
		def not_zero():
			def memcon_s():
				return mem_structure_im.copy()
			def memcon_d():
				return mem_structure_om.copy()
			mem_structure = jax.lax.cond(x > 0,memcon_d,memcon_s)
			return self.smj(-jnp.abs(x)+zdist,grad,bead_num,mem_structure)
		def zero():
			return self.smj(-jnp.abs(x)+zdist,grad,bead_num,mem_structure_im)
		return jax.lax.cond(not self.dbmem,zero,not_zero)
		
	#This function plots a cross section of the potential field
	def plot_potential(self,bead_num):
		ms_im = self.mem_structure_im
		ms_om = self.mem_structure_om
		zs = jnp.linspace(-30,30,500)
		xs = jnp.linspace(-30,30,10)
		vals = jnp.zeros((500,10))
		def plot_loop(vals,ind):
			vals = vals.at[ind].set(self.dmsj(zs[ind],0,2,bead_num,ms_im,ms_om))
			return vals,ind
		vals,_ = jax.lax.scan(plot_loop,vals,jnp.arange(500))
		plt.imshow(vals,extent = (-30,30,-30,30))
		plt.colorbar()
		plt.show()		
		
	#We define a smooth value indicating if a bead is in the membrane
	@jax.jit
	def sibj(self,x,grad,mem_structure):
		bd1 = sj(x-mem_structure[0],grad)
		bd2 = (1.0-sj(x-mem_structure[7],grad))*bd1
		return bd2
		

	
	#A function to calculate the potential of a bead at a given z position
	#This is done by calculating the potential of each free direction 
	@jax.jit
	def calc_pot_at_z_jit(self,z,zdist,ball,bead_type,curv,mem_structure_im,mem_structure_om):	
		plmi = jnp.sign(curv)		
		smoothness = 2
		tot_pot_count = jnp.zeros(2)
		def calc_pot_fun_1(tot_pot_count,ind):
			def pot_cond_0(tot_pot_count):
				return tot_pot_count
			def not_pot_cond_0(tot_pot_count):
				def zero_curv():
					cenz1 = 1e6
					cenz2 = -1e6
					cen1 = jnp.array([0.0,0.0,cenz1])
					cen2 = jnp.array([0.0,0.0,cenz2])
					circ_pos = ball[ind,:]+z[:]
					zpos1 = -(jnp.linalg.norm(circ_pos-cen1)-jnp.abs(cenz1))
					zpos2 = jnp.linalg.norm(circ_pos-cen2)-jnp.abs(cenz2)		
					tot_pot1 = self.surf_b_vals[ind]*self.dmsj(zpos1,zdist,smoothness,bead_type,mem_structure_im,mem_structure_om)
					tot_pot2 = self.surf_b_vals[ind]*self.dmsj(zpos2,zdist,smoothness,bead_type,mem_structure_im,mem_structure_om)
					adj_tot_pot = ((tot_pot2-tot_pot1)/2e-6)*curv+(tot_pot1+tot_pot2)/2
					return adj_tot_pot
				def nzero_curv():
					cenz = -1.0/(curv+plmi*1e-9)
					cen = jnp.array([0.0,0.0,cenz])
					circ_pos = ball[ind,:]+z[:]
					zpos = jnp.sign(curv)*(jnp.linalg.norm(circ_pos-cen)-jnp.abs(cenz))
					adj_tot_pot = self.surf_b_vals[ind]*self.dmsj(zpos,zdist,smoothness,bead_type,mem_structure_im,mem_structure_om)
					return adj_tot_pot
				adj_tot_pot = jax.lax.cond(jnp.abs(curv) < 1e-6,zero_curv,nzero_curv)				
				
				
				
				tot_pot_count = tot_pot_count.at[1].set(tot_pot_count[1]+1)
				tot_pot_count = tot_pot_count.at[0].set(tot_pot_count[0]+adj_tot_pot)
				return tot_pot_count
			tot_pot_count = jax.lax.cond(jnp.linalg.norm(ball[ind]) < 1e-5,pot_cond_0,not_pot_cond_0,tot_pot_count)
			return tot_pot_count,ind
		tot_pot_count,_ = jax.lax.scan(calc_pot_fun_1,tot_pot_count,jnp.arange(ball.shape[0]))
		tot_pot = tot_pot_count[0]/(tot_pot_count[1]+1e-5)
		return tot_pot
		
	#This is another function to polt the potential but for differing curvatures
	def show_pfield(self,bead_type,curv):
		xx = jnp.linspace(-41,39,100)
		zz = jnp.linspace(-40,40,100)
		cgrid = jnp.zeros((100,100))
		def spf1(cgrid,ind):
			ind2 = ind
			def spf2(cgrid,ind):
				grid_pos = jnp.array([xx[ind],0,zz[ind2]])
				cgrid = cgrid.at[ind2,ind].set(self.calc_pot_at_z_jit(grid_pos,0,jnp.array([[1,0,0]]),bead_type,curv,self.mem_structure_im,self.mem_structure_om))
				return cgrid,ind
			cgrid, _ = jax.lax.scan(spf2,cgrid,jnp.arange(100))
			return cgrid,ind
		cgrid,_ = jax.lax.scan(spf1,cgrid,jnp.arange(100))
		plt.imshow(cgrid)
		plt.show()
		
		
	#This is function fot calculating the potential of the PG layer for a bead at a z value
	@jax.jit
	def pg_pot_at_z(self,z,ball,bead_type):
		smoothness = 2
		tot_pot_count = jnp.zeros(2)
		def calc_pot_fun_1(tot_pot_count,ind):
			def pot_cond_0(tot_pot_count):
				return tot_pot_count
			def not_pot_cond_0(tot_pot_count):
				zpos = ball[ind,2]+z
				zpos_abs = jnp.abs(zpos)
				tot_pot_count = tot_pot_count.at[1].set(tot_pot_count[1]+1)
				tot_pot_count = tot_pot_count.at[0].set(tot_pot_count[0]+self.pg_pot(zpos,smoothness,bead_type))
				return tot_pot_count
			tot_pot_count = jax.lax.cond(jnp.linalg.norm(ball[ind]) < 1e-5,pot_cond_0,not_pot_cond_0,tot_pot_count)
			return tot_pot_count,ind
		tot_pot_count,_ = jax.lax.scan(calc_pot_fun_1,tot_pot_count,jnp.arange(ball.shape[0]))
		tot_pot = tot_pot_count[0]/(tot_pot_count[1]+1e-5)
		return tot_pot
	
	#This is a more accurate calculation of the electrostatic potentail contribution. Currently not in use, as it may not be more accurate. Further 
	#investigation is required to find the best method of calculating charge contributions.	
	@jax.jit
	def new_charge_diff(self,z,zdist,curv,bead_type,mem_structure_im,mem_structure_om,c_scal):
		charge_const = -92.64 #This is hard coded calculated such that units line up with paramters from martini
		charge_const_im = charge_const*(self.charge_mult*c_scal)*self.Charge_B_mins[bead_type]
		charge_const_om = charge_const*(self.charge_mult_om*c_scal)*self.Charge_B_mins[bead_type]
		tot_charge = 0
		zpos = z+zdist
		grid_ex = 11
		grid_nums = 50
		cut_width = 0.01
		dx = jnp.pi*grid_ex*grid_ex/grid_nums

		def in_charge(x,msa,msb):
			return x
		def not_in_charge(x,msa,msb):
			def above(x,msa,msb):
				retval = msa
				return retval
			def below(x,msa,msb):
				retval = msb
				return retval
			retval = jax.lax.cond(x>msa,above,below,x,msa,msb)
			return retval
		zposa = jax.lax.cond((zpos<mem_structure_im[7])*(zpos>mem_structure_im[0]),in_charge,not_in_charge,zpos,mem_structure_im[7],mem_structure_im[0])
		charge_val = 0
		
		def zero_curv():
			return create_disk_grid_charg(grid_nums,grid_ex)
		def nzero_curv():
			orad = (1/(jnp.abs(curv)+1e-9))+zposa
			end_ang = jnp.arcsin(grid_ex/orad)/jnp.pi
			grid = create_sph_grid_charg(grid_nums,end_ang,orad)
			return grid
		grid = jax.lax.cond(jnp.abs(curv)< 1e-6,zero_curv,nzero_curv)	
		def calc_charge_fun(tot_charge,ind):
			imb = 1e-5
			zpos = z+tot_charge[2]
			point_a = jnp.array([0.0,0.0,zpos])
			point_b = jnp.array([grid[ind,0],grid[ind,1],grid[ind,2]+tot_charge[1]+1e-5])
			dist = jnp.linalg.norm(point_a-point_b)
			def not_cutoff(tot_charge):
				def lfive(tot_charge):
					tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*((tot_charge[3]+tot_charge[1]*imb)/5.0))
					return tot_charge
				def gfive(tot_charge):
					tot_charge = tot_charge.at[0].set(tot_charge[0]+shifted_cos_cutoff(dist-grid_ex+cut_width,cut_width)*dx*((tot_charge[3]+tot_charge[1]*imb)/(jnp.abs(dist))))
					return tot_charge
				tot_charge = jax.lax.cond(dist < 5.0, lfive,gfive,tot_charge)
				return tot_charge
			def cutoff(tot_charge):
				return tot_charge
			tot_charge = jax.lax.cond(dist > grid_ex,cutoff,not_cutoff,tot_charge)
			return tot_charge,ind
		retval,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa,zdist,charge_const_im-1e-5]),jnp.arange(grid_nums))
		#retvalb,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposb,zdist,charge_const_im]),jnp.arange(grid_nums))
		charge_val = retval[0]#+retvalb[0]
		def zero(charge_val):
			return charge_val
		def not_zero(charge_val):
			zpos = z-zdist
			zposa = jax.lax.cond((zpos<mem_structure_om[7])*(zpos>mem_structure_om[0]),in_charge,not_in_charge,zpos,mem_structure_om[7],mem_structure_om[0])
			retvalc,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa,-zdist,charge_const_om-1e-5]),jnp.arange(grid_nums))
			
			charge_val += retvalc[0]# + retvald[0]
			
			return charge_val
		charge_val = jax.lax.cond(not self.dbmem,zero,not_zero,charge_val)
		return charge_val
	#This is the currently in use charge function. As noted before this may not be the most accurate and further investigation is required.
	@jax.jit
	def new_charge(self,z,zdist,curv,bead_type,mem_structure_im,mem_structure_om,c_scal):		
		charge_const = -92.64 #This is hard coded calculated such that units line up with paramters from martini
		charge_const_im = charge_const*(self.charge_mult)*self.Charge_B_mins[bead_type]
		charge_const_om = charge_const*(self.charge_mult_om)*self.Charge_B_mins[bead_type]
		tot_charge = 0
		zpos = z+zdist
		grid_ex = 11
		grid_nums = 50
		dx = jnp.pi*grid_ex*grid_ex/grid_nums
		def in_charge(x,msa,msb):
			return x
		def not_in_charge(x,msa,msb):
			def above(x,msa,msb):
				retval = msa
				return retval
			def below(x,msa,msb):
				retval = msb
				return retval
			retval = jax.lax.cond(x>msa,above,below,x,msa,msb)
			return retval
		zposa = jax.lax.cond((zpos<mem_structure_im[3])*(zpos>mem_structure_im[0]),in_charge,not_in_charge,zpos,mem_structure_im[3],mem_structure_im[0])
		zposb = jax.lax.cond((zpos<mem_structure_im[7])*(zpos>mem_structure_im[4]),in_charge,not_in_charge,zpos,mem_structure_im[7],mem_structure_im[4])
		
		def zero_curv():
			return create_disk_grid_charg(grid_nums,grid_ex),1.0
		def nzero_curva():
			orad = (1/(jnp.abs(curv)+1e-9))+jnp.sign(curv)*zposa
			toobig = jnp.min(jnp.array([1,grid_ex/orad]))
			end_ang = (grid_nums+0.5)/(1-jnp.sqrt(1-toobig*toobig))
			
			
			acc_end_ang = jnp.arccos(1-(grid_nums+0.5)/end_ang)
			area_c = 2*jnp.pi*orad*orad*(1-jnp.cos(acc_end_ang))
			area_mult = area_c/(jnp.pi*grid_ex*grid_ex)		
			grid = jnp.sign(curv)*create_sph_grid_charg(grid_nums,end_ang,orad)
			return grid,area_mult
		def nzero_curvb():
			orad = (1/(jnp.abs(curv)+1e-9))+jnp.sign(curv)*zposb
			toobig = jnp.min(jnp.array([1,grid_ex/orad]))
			end_ang = (grid_nums+0.5)/(1-jnp.sqrt(1-toobig*toobig))
			acc_end_ang = jnp.arccos(1-(grid_nums+0.5)/end_ang)
			area_c = 2*jnp.pi*orad*orad*(1-jnp.cos(acc_end_ang))
			area_mult = area_c/(jnp.pi*grid_ex*grid_ex)		
			
			grid = jnp.sign(curv)*create_sph_grid_charg(grid_nums,end_ang,orad)
			return grid,area_mult
		grid_a,aream_a = jax.lax.cond(jnp.abs(curv)< 1e-6,zero_curv,nzero_curva)			
		grid_b,aream_b = jax.lax.cond(jnp.abs(curv)< 1e-6,zero_curv,nzero_curvb)	
		charge_val = 0
		acmin = c_scal*1e-2*self.Charge_B_mins[bead_type]
		bcmin = 0
		def calc_charge_fun(carry,ind):
			tot_charge,grid=carry
			zpos = z+tot_charge[2]
			point_a = jnp.array([0.0,0.0,zpos])
			point_b = jnp.array([grid[ind,0],grid[ind,1],grid[ind,2]+tot_charge[1]+1e-5])
			dist = jnp.linalg.norm(point_a-point_b)
			def not_cutoff(tot_charge):
				def lfive(tot_charge):
					tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(tot_charge[3]/5.0-tot_charge[3]/grid_ex))
					return tot_charge
				def gfive(tot_charge):
					tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(tot_charge[3]/(jnp.abs(dist))-tot_charge[3]/grid_ex))
					return tot_charge
				tot_charge = jax.lax.cond(dist < 5, lfive,gfive,tot_charge)
				return tot_charge
			def cutoff(tot_charge):
				return tot_charge
			tot_charge = jax.lax.cond(dist > grid_ex,cutoff,not_cutoff,tot_charge)
			return (tot_charge,grid),ind
		(retval,_),_=jax.lax.scan(calc_charge_fun,(jnp.array([0,zposa,zdist,charge_const_im-acmin]),grid_a),jnp.arange(grid_nums))
		(retvalb,_),_=jax.lax.scan(calc_charge_fun,(jnp.array([0,zposb,zdist,charge_const_im-bcmin]),grid_b),jnp.arange(grid_nums))
		charge_val = retval[0]*aream_a+retvalb[0]*aream_b
		def zero(charge_val):
			return charge_val
		def not_zero(charge_val):
			zpos = z-zdist
			grid_temp = create_disk_grid_charg(grid_nums,grid_ex)
			zposa = jax.lax.cond((zpos<mem_structure_om[3])*(zpos>mem_structure_om[0]),in_charge,not_in_charge,zpos,mem_structure_om[3],mem_structure_om[0])
			zposb = jax.lax.cond((zpos<mem_structure_om[7])*(zpos>mem_structure_om[4]),in_charge,not_in_charge,zpos,mem_structure_om[7],mem_structure_om[4])
			(retvalc,_),_=jax.lax.scan(calc_charge_fun,(jnp.array([0,zposa,-zdist,charge_const_om-acmin]),grid_temp),jnp.arange(grid_nums))
			(retvald,_),_=jax.lax.scan(calc_charge_fun,(jnp.array([0,zposb,-zdist,charge_const_om-bcmin]),grid_temp),jnp.arange(grid_nums))
		
			charge_val += retvalc[0] + retvald[0]
			
			return charge_val
		charge_val = jax.lax.cond(not self.dbmem,zero,not_zero,charge_val)#?
		return charge_val
		
	#As above but for the PG layer
	@jax.jit
	def pg_charge_fun(self,z,bead_type):
		charge_const = -92.64
		charge_const = charge_const*(self.pg_charge)*self.Charge_B_mins[bead_type]
		tot_charge = 0
		zpos = z
		grid_ex = 11
		grid_nums = 50
		dx = jnp.pi*grid_ex*grid_ex/grid_nums
		grid = 	create_disk_grid(grid_nums,grid_ex)
		def in_charge(x,msa,msb):
			return x
		def not_in_charge(x,msa,msb):
			def above(x,msa,msb):
				retval = msa
				return retval
			def below(x,msa,msb):
				retval = msb
				return retval
			retval = jax.lax.cond(x>msa,above,below,x,msa,msb)
			return retval
		zposa = jax.lax.cond((zpos<self.pg_thickness/2)*(zpos>-self.pg_thickness/2),in_charge,not_in_charge,zpos,self.pg_thickness/2,-self.pg_thickness/2)
		charge_val = 0
		def calc_charge_fun(tot_charge,ind):
			zpos = z+tot_charge[2]
			point_a = jnp.array([0.0,0.0,zpos])
			point_b = jnp.array([grid[ind,0]*jnp.cos(grid[ind,1]),grid[ind,0]*jnp.sin(grid[ind,1]),tot_charge[1]+1e-5])
			dist = jnp.linalg.norm(point_a-point_b)
			def not_cutoff(tot_charge):
				def lfive(tot_charge):
					tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(charge_const/5.0-charge_const/grid_ex))
					return tot_charge
				def gfive(tot_charge):
					tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(charge_const/(jnp.abs(dist))-charge_const/grid_ex))
					return tot_charge
				tot_charge = jax.lax.cond(dist < 5, lfive,gfive,tot_charge)
				return tot_charge
			def cutoff(tot_charge):
				return tot_charge
			tot_charge = jax.lax.cond(dist > grid_ex,cutoff,not_cutoff,tot_charge)
			return tot_charge,ind
		retval,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa]),jnp.arange(5))
		charge_val = retval[0]
		return charge_val
			
					
	#A function that calculates if a protein is fully ejected from the membrane
	@jax.jit
	def calc_in_water_jit(self,position,tol,mem_structure_im,mem_structure_om):
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		curv = position[7]/10
		ang1 /= jnp.linalg.norm(ang1)
		plmi = jnp.sign(curv)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			def zero_curv(tot_pot):
				cenz1 = 1e6
				cenz2 = -1e6
				cen1 = jnp.array([0.0,0.0,cenz1])
				cen2 = jnp.array([0.0,0.0,cenz2])
				circ_pos = tester_poses[ind,:]
				zpos1 = -(jnp.linalg.norm(circ_pos-cen1)-jnp.abs(cenz1))
				zpos2 = jnp.linalg.norm(circ_pos-cen2)-jnp.abs(cenz2)		
				tot_pot1 = self.surf_b_vals[ind]*self.new_charge(zpos1,zdist,-1e-6,self.bead_types[ind],mem_structure_im,mem_structure_om,1)
				tot_pot2 = self.surf_b_vals[ind]*self.new_charge(zpos2,zdist,1e-6,self.bead_types[ind],mem_structure_im,mem_structure_om,1)
				adj_tot_pot = ((tot_pot2-tot_pot1)/2e-6)*curv+(tot_pot1+tot_pot2)/2
				tot_pot += adj_tot_pot
				return tot_pot
			def nzero_curv(tot_pot):
				cenz = -1.0/(curv+plmi*1e-9)
				cen = jnp.array([0.0,0.0,cenz])
				circ_pos = tester_poses[ind,:]
				zpos = jnp.sign(curv)*(jnp.linalg.norm(circ_pos-cen)-jnp.abs(cenz))
				tot_pot += self.surf_b_vals[ind]*self.new_charge(zpos,zdist,curv,self.bead_types[ind],mem_structure_im,mem_structure_om,1)
				return tot_pot
			tot_pot = jax.lax.cond(jnp.abs(curv) < 1e-6,zero_curv,nzero_curv,tot_pot)
			tot_pot += self.surf_b_vals[ind]*self.calc_pot_at_z_jit(tester_poses[ind,:],zdist,test_spheres[ind],self.bead_types[ind],curv,mem_structure_im,mem_structure_om)
			
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		return tot_pot<tol


	#A function that calculates the potential of the protein for a given position
	#@jax.jit			
	def calc_pot_jit(self,position,mem_structure_im,mem_structure_om,c_scal):
		disp_penalty = 0.025
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		curv = position[7]/10
		plmi = jnp.sign(curv)
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0.0
		def calc_pot_fun_1(tot_pot,ind):
			def zero_curv(tot_pot):
				cenz1 = 1e6
				cenz2 = -1e6
				cen1 = jnp.array([0.0,0.0,cenz1])
				cen2 = jnp.array([0.0,0.0,cenz2])
				circ_pos = tester_poses[ind,:]
				zpos1 = -(jnp.linalg.norm(circ_pos-cen1)-jnp.abs(cenz1))	
				zpos2 = jnp.linalg.norm(circ_pos-cen2)-jnp.abs(cenz2)		
				tot_pot1 = self.surf_b_vals[ind]*self.new_charge(zpos1,zdist,-1e-6,self.bead_types[ind],mem_structure_im,mem_structure_om,c_scal)
				tot_pot2 = self.surf_b_vals[ind]*self.new_charge(zpos2,zdist,1e-6,self.bead_types[ind],mem_structure_im,mem_structure_om,c_scal)		
				adj_tot_pot = ((tot_pot2-tot_pot1)/2e-6)*curv+(tot_pot1+tot_pot2)/2
				tot_pot += adj_tot_pot
				return tot_pot
			def nzero_curv(tot_pot):
				cenz = -1.0/(curv+plmi*1e-9)
				cen = jnp.array([0.0,0.0,cenz])
				circ_pos = tester_poses[ind,:]
				zpos = jnp.sign(curv)*(jnp.linalg.norm(circ_pos-cen)-jnp.abs(cenz))
				tot_pot += self.surf_b_vals[ind]*self.new_charge(zpos,zdist,curv,self.bead_types[ind],mem_structure_im,mem_structure_om,c_scal)
				return tot_pot
			tot_pot = jax.lax.cond(jnp.abs(curv) < 1e-6,zero_curv,nzero_curv,tot_pot)
			tot_pot += self.surf_b_vals[ind]*self.calc_pot_at_z_jit(tester_poses[ind,:],zdist,test_spheres[ind],self.bead_types[ind],curv,mem_structure_im,mem_structure_om)
			
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		return tot_pot + 10000*jnp.abs(curv)*sj(jnp.abs(curv)-0.02,10000)
		
		
		
	#This function calulates the potential as a vector of each bead's contribution. This is used when writing this info to PDB file.
	#It is particularially useful when trying to understand why certain orientations are not working.
	@jax.jit			
	def calc_pot_per_bead(self,position,mem_structure_im,mem_structure_om):
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		curv = position[7]/10
		plmi = jnp.sign(curv)
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = jnp.zeros(tester_poses.shape[0])
		def calc_pot_fun_1(tot_pot,ind):
			def zero_curv(tot_pot):
				cenz1 = 1e6
				cenz2 = -1e6
				cen1 = jnp.array([0.0,0.0,cenz1])
				cen2 = jnp.array([0.0,0.0,cenz2])
				circ_pos = tester_poses[ind,:]
				zpos1 = -(jnp.linalg.norm(circ_pos-cen1)-jnp.abs(cenz1))	
				zpos2 = jnp.linalg.norm(circ_pos-cen2)-jnp.abs(cenz2)		
				tot_pot1 = self.surf_b_vals[ind]*self.new_charge(zpos1,zdist,-1e-6,self.bead_types[ind],mem_structure_im,mem_structure_om,1)
				tot_pot2 = self.surf_b_vals[ind]*self.new_charge(zpos2,zdist,1e-6,self.bead_types[ind],mem_structure_im,mem_structure_om,1)
				adj_tot_pot = ((tot_pot2-tot_pot1)/2e-6)*curv+(tot_pot1+tot_pot2)/2
				tot_pot = tot_pot.at[ind].set(tot_pot[ind]+adj_tot_pot)
				return tot_pot
			def nzero_curv(tot_pot):
				cenz = -1.0/(curv+plmi*1e-9)
				cen = jnp.array([0.0,0.0,cenz])
				circ_pos = tester_poses[ind,:]
				zpos = jnp.sign(curv)*(jnp.linalg.norm(circ_pos-cen)-jnp.abs(cenz))
				tot_pot = tot_pot.at[ind].set(tot_pot[ind]+self.surf_b_vals[ind]*self.new_charge(zpos,zdist,curv,self.bead_types[ind],mem_structure_im,mem_structure_om,1))
				return tot_pot
			tot_pot = jax.lax.cond(jnp.abs(curv) < 1e-6,zero_curv,nzero_curv,tot_pot)
			tot_pot = tot_pot.at[ind].set(tot_pot[ind]+self.surf_b_vals[ind]*self.calc_pot_at_z_jit(tester_poses[ind,:],zdist,test_spheres[ind],self.bead_types[ind],curv,mem_structure_im,mem_structure_om))
			
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		return 100.0*tot_pot
		
	#Function to calculate the potential of the PG layer at a given z
	@jax.jit			
	def calc_pot_pg(self,position,pg_z,mem_structure_im,mem_structure_om):
		charge_const = -92.64
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			tot_pot += self.surf_b_vals[ind]*self.pg_pot_at_z(tester_poses[ind,2]+pg_z,test_spheres[ind],self.bead_types[ind])
			tot_pot += self.surf_b_vals[ind]*self.pg_charge_fun(tester_poses[ind,2]+pg_z,self.bead_types[ind])
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		tot_pot += potential_between_sheets(zdist-pg_z+mem_structure_om[0]-self.pg_thickness/2,10,20,-self.sheet_charge*charge_const)+potential_between_sheets(zdist+pg_z+mem_structure_im[0]-self.pg_thickness/2,10,20,-self.sheet_charge*charge_const)
		return tot_pot
		

	#Differentiating the potential to get gradients for minimisation
	@jax.jit
	def pot_grad(self,position,c_scal):
		return jax.grad(self.calc_pot_jit,argnums=0)(position,self.mem_structure_im,self.mem_structure_om,c_scal)



	#The following functions are involved in finding the hydrphobic core of the protein
	def get_hydro_core_v3(self):
		numer = self.surface_poses.shape[0]
		sposes = self.surface_poses
		hydro_val = jnp.array(self.W_B_mins)
		hydro_index = jnp.array(jax.device_get(self.bead_types),dtype=int)
		
		
		@jax.vmap
		def gch1(ind):
			ind_fix = ind
			mens = jnp.zeros(numer+1)
			def gch2(mens,ind):
				def in_r(mens):
					vnd = jnp.array(mens[-1],dtype = int)
					mens = mens.at[vnd].set(hydro_val[hydro_index[ind]])
					mens = mens.at[-1].set(mens[-1] + 1)
					return mens
				def nin_r(mens):
					return mens
				mens = jax.lax.cond(jnp.linalg.norm(sposes[ind]-sposes[ind_fix]) < 20,in_r,nin_r,mens)
				return mens,ind
			mens,_ = jax.lax.scan(gch2,mens,jnp.arange(numer))
			vnd = jnp.array(mens[-1],dtype = int)
			nnh = jnp.sum(mens[:-1])/vnd
			return nnh
		nnh = gch1(jnp.arange(numer))
		cut = jnp.percentile(nnh,98)
		core = jnp.mean(sposes[nnh>=cut],axis=0)
		return core
		
	def get_hydro_core_v3_p1(self,zdir,xydir):
		sposes = position_pointv2_jit(0,zdir,xydir,self.surface_poses)
		numer = self.surface_poses.shape[0]
		hydro_val = jnp.array(self.W_B_mins)
		hydro_index = jnp.array(jax.device_get(self.bead_types),dtype=int)
		
		@jax.vmap
		def gch1(ind):
			ind_fix = ind
			mens = jnp.zeros(numer+1)
			def gch2(mens,ind):
				def in_r(mens):
					vnd = jnp.array(mens[-1],dtype = int)
					mens = mens.at[vnd].set(hydro_val[hydro_index[ind]])
					mens = mens.at[-1].set(mens[-1] + 1)
					return mens
				def nin_r(mens):
					return mens
				mens = jax.lax.cond(jnp.linalg.norm(sposes[ind]-sposes[ind_fix]) < 20,in_r,nin_r,mens)
				return mens,ind
			mens,_ = jax.lax.scan(gch2,mens,jnp.arange(numer))
			vnd = jnp.array(mens[-1],dtype = int)
			nnh = jnp.sum(mens[:-1])/vnd
			return nnh
		nnh = gch1(jnp.arange(numer))
		cut = jnp.percentile(nnh,98)
		core = jnp.mean(sposes[nnh>=cut],axis=0)
		return core
	
	#A function that takes a weighted mean of points to try and get a good starting guess for insertion depth
	@jax.jit
	def get_hydrophobic_core_jit(self):
		hydro_cut_off = -3
		hydro_range = 10
		av_pos_count = jnp.zeros(4)
		def hydro_fun_1(carry,ind):
			av_countp = 0
			def is_hydro_1(av_countp):
				def hydro_fun2(ind_fix,ind):
					def is_close(av_countpp):
						def is_hydro_2(av_countpp):
							av_countpp += 1
							return av_countpp
						def not_hydro_2(av_countpp):
							return av_countpp
						av_countpp = jax.lax.cond(self.surf_b_vals[ind]*self.W_B_mins[self.bead_types[ind]] > hydro_cut_off,is_hydro_2,not_hydro_2,av_countpp)
						return av_countpp
					def not_close(av_countpp):
						return av_countpp
					ind_fix = ind_fix.at[1].set(jax.lax.cond(jnp.linalg.norm(self.surface_poses[ind]-self.surface_poses[ind_fix[0]]) < hydro_range,is_close,not_close,ind_fix[1]))
					return ind_fix,ind
				out,_ = jax.lax.scan(hydro_fun2,jnp.array([ind,av_countp]),jnp.arange(self.surface_poses.shape[0]))
				av_countp = out[1]
				return av_countp
			def not_hydro_1(av_countp):				
				return av_countp
			av_countp = jax.lax.cond(self.surf_b_vals[ind]*self.W_B_mins[self.bead_types[ind]] > hydro_cut_off,is_hydro_1,not_hydro_1,av_countp)
			return carry,av_countp
		_,all_count = jax.lax.scan(hydro_fun_1,0,jnp.arange(self.surface_poses.shape[0]))
		lower_bound = jnp.max(all_count)*0.9
		
		def hydro_fun_3(av_pos_count,ind):
			def is_dens(av_pos_count):
				av_pos_count = av_pos_count.at[3].set(av_pos_count[3]+all_count[ind]*all_count[ind])
				av_pos_count = av_pos_count.at[:3].set(av_pos_count[:3]+all_count[ind]*all_count[ind]*self.surface_poses[ind])
				return av_pos_count
			def is_not_dens(av_pos_count):
				return av_pos_count
			av_pos_count = jax.lax.cond(all_count[ind] >=lower_bound,is_dens,is_not_dens,av_pos_count)
			return av_pos_count,ind
		av_pos_count,_ = jax.lax.scan(hydro_fun_3,av_pos_count,jnp.arange(self.surface_poses.shape[0]))
		return av_pos_count[:3]/av_pos_count[3]
		

	#An improved version of the above
	def hydro_core_imp(self,surface_poses,bead_types):
		num_binsx = jnp.array(jnp.floor(jnp.max(jnp.abs(surface_poses[:,0]))/10),dtype=int)
		num_binsy = jnp.array(jnp.floor(jnp.max(jnp.abs(surface_poses[:,1]))/10),dtype=int)
		num_binsz = jnp.array(jnp.floor(jnp.max(jnp.abs(surface_poses[:,2]))/10),dtype=int)
		rangerx = jnp.max(surface_poses[:,0])
		rangery = jnp.max(surface_poses[:,1])
		rangerz = jnp.max(surface_poses[:,2])
		x = jnp.linspace(-rangerx,rangerx,num_binsx)
		y = jnp.linspace(-rangery,rangery,num_binsy)
		z = jnp.linspace(-rangerz,rangerz,num_binsz)
		xm,ym,zm = jnp.meshgrid(x,y,z)
		ys = jnp.zeros((num_binsx,num_binsy,num_binsz,2))
		def hydro_fingerprint_fun(ys,ind):
			xpos = surface_poses[ind][0]
			ypos = surface_poses[ind][1]
			zpos = surface_poses[ind][2]
			hydro_val =self.W_B_mins[bead_types[ind]]
			x_ind = jnp.array(num_binsx*(xpos+rangerx)/(rangerx*2),dtype=int)
			y_ind = jnp.array(num_binsy*(ypos+rangery)/(rangery*2),dtype=int)
			z_ind = jnp.array(num_binsz*(zpos+rangerz)/(rangerz*2),dtype=int)
			ys = ys.at[x_ind,y_ind,z_ind,0].set(ys[x_ind,y_ind,z_ind,0]+hydro_val)
			ys = ys.at[x_ind,y_ind,z_ind,1].set(ys[x_ind,y_ind,z_ind,1]+1)
			return ys,ind
		ys,_ = jax.lax.scan(hydro_fingerprint_fun,ys,jnp.arange(surface_poses.shape[0]))
		y_vals = ys[:,:,:,0]/(ys[:,:,:,1]+1e-5)
		y_vals = y_vals.ravel()
		xmf = xm.ravel()
		ymf = ym.ravel()
		zmf = zm.ravel()
		def hydro_norm(y_vals,ind):
			def zero(y_vals):
				y_vals = y_vals.at[ind].set(-8)
				return y_vals
			def not_zero(y_vals):
				return y_vals
			y_vals = jax.lax.cond(y_vals[ind] == 0,zero,not_zero,y_vals)
			return y_vals,ind
		y_vals,_ = jax.lax.scan(hydro_norm,y_vals,jnp.arange(y_vals.shape[0]))
		is_hydro = xmf[y_vals>-3].size
		hydro_core = jnp.zeros(3)
		hydro_corex = jnp.mean(xmf[y_vals>-3])
		hydro_corey = jnp.mean(ymf[y_vals>-3])
		hydro_corez = jnp.mean(zmf[y_vals>-3])
		def empty():
			return jnp.mean(surface_poses,axis=0)
		def not_empty():
			return jnp.array([hydro_corex,hydro_corey,hydro_corez])
		hydro_core = jax.lax.cond(is_hydro == 0,empty,not_empty)
		return hydro_core
		
	#A simple function to find the extend of the protein in certain configuration
	def get_extent(self,position):
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		return jnp.min(tester_poses[:,0]),jnp.max(tester_poses[:,0]),jnp.min(tester_poses[:,1]),jnp.max(tester_poses[:,1]),jnp.min(tester_poses[:,2]),jnp.max(tester_poses[:,2])

	#A function to finds the minimal insertion depth (within range) via grid search
	#This is in general expensive but is good for peripheral proteins
	def z_pot(self,max_iter,ranger,zdir,xydir,shift = 0):
		zs = jnp.linspace(-ranger+shift,ranger+shift,max_iter)#20
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(8)
			posi = posi.at[1].set(zs[ind])
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			posi = posi.at[7].set(0)
			pots = pots.at[ind].set(self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om,0))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		return zs[best_ind]
	
	#A series of function for getting potential curves while varying certain variables
	def z_pot_for_ps(self,max_iter,ranger,zdir,xydir,mem_structure,shift = 0):
		zs = jnp.linspace(-ranger+shift,ranger+shift,max_iter)
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(8)
			posi = posi.at[1].set(zs[ind])
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			posi = posi.at[7].set(0)
			pots = pots.at[ind].set(self.calc_pot_jit(posi,mem_structure,mem_structure,0))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		return zs[best_ind]
		
	def z_pot_no_range(self,max_iter,zdir,xydir):
		posi = jnp.zeros(8)
		posi = posi.at[2:5].set(zdir)
		posi = posi.at[5:7].set(xydir)
		posi = posi.at[7].set(0)
		_,_,_,_,mini,maxi = self.get_extent(posi)
		zs = jnp.linspace(mini-(self.memt_tails/2-10),maxi+(self.memt_tails/2-10),max_iter)#20 TODO Want it to sink into membrane a little more probably but need to go so test later <------------------------------------------------------------------
		def calc_pot_fun(posi,ind):
			posi = posi.at[1].set(zs[ind])
			reter = self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om,0)
			return posi,reter
		_,pots = jax.lax.scan(calc_pot_fun,posi,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		return zs[best_ind]
		
		
	#The same as above but it plots the potential (for debugging purposes)
	def z_pot_plot(self,max_iter,ranger,zdir,xydir):
		posi = jnp.zeros(8)
		posi = posi.at[2:5].set(zdir)
		posi = posi.at[5:7].set(xydir)
		posi = posi.at[7].set(0)
		_,_,_,_,mini,maxi = self.get_extent(posi)
		rang_min = mini-(self.memt_tails/2+20)
		rang_max = maxi+(self.memt_tails/2+20)
		zs = jnp.linspace(rang_min,rang_max,max_iter)		
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(0)
			posi = posi.at[1].set(zs[ind])
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			posi = posi.at[7].set(0)
			pots = pots.at[ind].set(self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om,0))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		plt.plot(zs,pots)
		plt.ylabel("Potential energy")
		plt.xlabel("Z")
		plt.title("Potential energy against z position relative to center of membrane")
		plt.tight_layout()
		plt.show()
		
	@partial(jax.jit,static_argnums=1)
	def c_pot_plot(self,max_iter,zdir,xydir,zss):
		posi = jnp.zeros(8)
		posi = posi.at[1].set(zss)
		posi = posi.at[2:5].set(zdir)
		posi = posi.at[5:7].set(xydir)
		posi = posi.at[7].set(0)
		cs = jnp.linspace(-0.18,0.18,max_iter)		
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(8)
			posi = posi.at[1].set(zss)
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			posi = posi.at[7].set(cs[ind])
			pots = pots.at[ind].set(self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om,0))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		return pots,cs
		
	def ang_pot_plot(self,max_iter,ranger,zdir,xydir):
		zs = jnp.linspace(-jnp.pi,jnp.pi,max_iter)		
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			rot_mat = jnp.array([[jnp.cos(zs[ind]),0,-jnp.sin(zs[ind])],[0,1,0],[jnp.sin(zs[ind]),0,jnp.cos(zs[ind])]])
			posi = jnp.zeros(7)
			posi = posi.at[1].set(-10)
			posi = posi.at[2:5].set(jnp.dot(rot_mat,zdir))
			posi = posi.at[5:7].set(xydir)
			pots = pots.at[ind].set(self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om,0))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		plt.plot(zs,pots)
		plt.ylabel("Potential energy")
		plt.xlabel("Radians")
		plt.title("Potential energy against angle")
		plt.tight_layout()
		plt.show()
		
		
	#This is same as above but does not plot. This is for minima testing
	@partial(jax.jit,static_argnums=1)
	def z_pot_graph(self,max_iter,ranger,zdir,xydir,in_depth,curv):
		
		posi = jnp.zeros(8)
		posi = posi.at[2:5].set(zdir)
		posi = posi.at[5:7].set(xydir)
		posi = posi.at[7].set(curv)
		_,_,_,_,mini,maxi = self.get_extent(posi)
		rang_min = mini-(self.memt_tails/2+20)
		rang_max = maxi+(self.memt_tails/2+20)
		zs = jnp.linspace(rang_min,rang_max,max_iter)
		in_depth_ind = ((in_depth-rang_min)/(rang_max-rang_min))*max_iter
		in_depth_ind = jnp.array(in_depth_ind,dtype=int) 
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(8)
			posi = posi.at[1].set(zs[ind])
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			posi = posi.at[7].set(curv)
			pots = pots.at[ind].set(self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om,0))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
			
		return pots,zs,in_depth_ind

	
	#This function plots the PG layer potential surface(curve)
	def pg_pot_plot(self,nums,min_ind,mem_structure_im,mem_structure_om):
		if(self.curva):
			mins = np.array(self.minima_c)
		else:
			mins = np.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		ranger = jnp.abs(min_poses[min_ind][0]*jnp.cos(min_poses[min_ind][3]))
		xs = jnp.linspace(-ranger-self.mem_structure[0]+self.pg_thickness/2,ranger+self.mem_structure[0]-self.pg_thickness/2,nums)
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		test_pos = jnp.concatenate([jnp.array([min_poses[min_ind][0],min_poses[min_ind][1]]),start_zdir,start_xydir])
		ys = [self.calc_pot_pg(test_pos,z,mem_structure_im,mem_structure_om) for z in xs]
		plt.plot(xs,ys)
		bot,top = plt.ylim()
		plt.ylim(0,3000)
		plt.show()
	
	#This function evaluates the position of the PG layer 
	def get_pg_pos(self,nums,min_ind,mem_structure_im,mem_structure_om,pg_guess):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		ranger = jnp.abs(min_poses[min_ind][0]*jnp.cos(min_poses[min_ind][3]))
		xs = jnp.linspace(-ranger-self.mem_structure[0]+self.pg_thickness/2,ranger+self.mem_structure[0]-self.pg_thickness/2,nums)
		pguess = jnp.zeros(xs.shape[0])
		def is_pgguess(pguess):
			xval = xs-(-ranger-self.mem_structure[0]+pg_guess)
			def pgloop(pguess,ind):
				pguess = pguess.at[ind].set(1.5*((0.2*xval[ind]*xval[ind]-50))*sj(-xval[ind]*xval[ind],0.03))
				return pguess,ind
			pguess,_ = jax.lax.scan(pgloop,pguess,jnp.arange(xs.shape[0]))		
			return pguess
		def isn_pgguess(pguess):
			return pguess
		pguess = jax.lax.cond(pg_guess > 0,is_pgguess,isn_pgguess,pguess)
			
				
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		test_pos = jnp.concatenate([jnp.array([min_poses[min_ind][0],min_poses[min_ind][1]]),start_zdir,start_xydir])
		ps = jnp.zeros(nums)
		def pg_pot_fun(ps,ind):
			ps = ps.at[ind].set(self.calc_pot_pg(test_pos,xs[ind],mem_structure_im,mem_structure_om))
			return ps, ind
		ps,_ = jax.lax.scan(pg_pot_fun,ps,jnp.arange(nums))
		return ps+pguess,xs
	

	#A good starting point for periplasmic spanning proteins is very important
	#This is as the potential energy landscape in the z direction has many minima
	#The current method used certainly biases the final position
	#However for these proteins the starting positions is known to be close already
	#Overall not much additional work is needed if the starting z pos is good.

	#A function that uses the hydrophobic core functions to determine a good starting periplasmic space width
	def calc_start_for_ps(self):
		hydro_core =  self.hydro_core_imp(self.surface_poses,self.bead_types)
		surface_poses = self.surface_poses - hydro_core
		bead_pos = self.bead_types[surface_poses[:,2]>0]
		bead_neg = self.bead_types[surface_poses[:,2]<0]
		surface_poses_pos = surface_poses[surface_poses[:,2]>0]
		surface_poses_neg = surface_poses[surface_poses[:,2]<0]
		
		pos_core = self.hydro_core_imp(surface_poses_pos,bead_pos)
		neg_core = self.hydro_core_imp(surface_poses_neg,bead_neg)
		zdist = (pos_core-neg_core)[2]/2
		start_z = (pos_core+neg_core)[2]/2
		return zdist,start_z
		
		 
	#A function that uses the z potemtail scanning functions to determine a good starting periplasmic space width
	def calc_start_for_ps_imp(self):
		maxz = jnp.max(self.surface_poses[:,2])+20
		minz = jnp.min(self.surface_poses[:,2])-20
		half_mem = self.memt_total/2.0
		rplus = jnp.abs(maxz/2)-half_mem
		rminus = jnp.abs(minz/2)-half_mem
		rangplus = rplus-rplus/4
		shiftplus = maxz/2+rplus/4
		rangminus = rminus-rminus/4
		shiftminus = minz/2-rminus/4
		charge_im_temp = self.charge_mult
		self.charge_mult =self.charge_mult_om
		zplus = self.z_pot_for_ps(100,rangplus,jnp.array([0,0,1]),jnp.array([1,0]),self.mem_structure_om,shift = shiftplus)
		self.charge_mult= charge_im_temp
		zminus = self.z_pot_for_ps(100,rangminus,jnp.array([0,0,1]),jnp.array([1,0]),self.mem_structure_im,shift = shiftminus)
		
		zdist = (zplus-zminus)/2
		start_z = (zplus+zminus)/2
		return zdist,start_z
		
	#A function the minimises potential from a given starting position
	def minimise(self,starting_pos,max_iter):	
		pot = 0
		data = jnp.zeros(25)
		num = 8
		gamma = jnp.array([self.gamma_val]*num,)
		ep = 1e-8
		tol = 1e-10
		data = data.at[:num].set(starting_pos[:num])
		decend = 0.0
		curv_decend = 0.0
		def zero(decend):
			return decend
		def not_zero(decend):
			decend = 1.0
			return decend
		decend = jax.lax.cond(not self.dbmem,zero,not_zero,decend)
		
		curv_decend = jax.lax.cond(not self.gcurv,zero,not_zero,curv_decend)		
		data = data.at[num-1].set(0)
		data = data.at[num*2:num*3].set(jnp.array([self.lr_pos]*num))#???
		data = data.at[num*2+num-1].set(data[num*2+num-1]/100000)
		data = data.at[num*3].set(1)

		def min_fun_1(data,ind):
			grad = jnp.zeros(num)
			timeser = 0			
			def go_g(grad):
				grad = jnp.array(self.pot_grad(data[:num],timeser))
				return grad
			def stop_g(grad):
				return grad
			grad = jax.lax.cond(data[num*3] > 0.5,go_g,stop_g,grad)
	
			def isnanj():
				return 1
			def isnnanj():
				return 0
			def close(data):
				data = data.at[num*3].set(0)
				return data
			def far(data):
				data = data.at[num*3].set(1)
				return data
			data = jax.lax.cond(jnp.linalg.norm(grad)<tol,close,far,data)	
			data = data.at[num:num*2].set(data[num:num*2]*gamma + (1-gamma)*grad*grad)
			

			rms_grad = jnp.sqrt(ep + data[num:num*2])
			rms_change = jnp.sqrt(ep+data[num*2:num*3])
			change = -(rms_change/rms_grad)*grad
			change = change.at[0].set(change[0]*decend)
			change = change.at[num-1].set(change[num-1]*curv_decend)
			
			
			data = data.at[:num].set(data[:num]+change)
			diff_data = jnp.array([data[0],-data[1],data[2],jnp.pi+data[3]])
			data = data.at[num*2:num*3].set(gamma*data[num*2:num*3]+(1-gamma)*change*change)
			return data,ind
		final_data,_ = jax.lax.scan(min_fun_1,data,jnp.arange(max_iter))
	
		final_pot = self.calc_pot_jit(final_data[:num],self.mem_structure_im,self.mem_structure_om,1)
		flipped_data = final_data.copy()
		
		#There were issue flipping the protein so a very explicit method is used (can be replaced to be something that is cleaner)
		normed = jnp.array([flipped_data[5],flipped_data[6],0.0])
		normed2 = jnp.array([flipped_data[2],flipped_data[3],flipped_data[4]])
		
		normed2 /= jnp.linalg.norm(normed2)
		
		direc1 = jnp.cross(normed2,jnp.array([0.0,0.0,1.0]))
		ang1 = (1-1e-8)*jnp.dot(normed2,jnp.array([0.0,0.0,1.0]))
		
		rot1qi = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
		rot1qi /= jnp.linalg.norm(rot1qi)
		rot1q = qcong_jit(rot1qi)

		
		xydir_cor = jnp.array([flipped_data[5],flipped_data[6],0.0])
		xydir_cor /= jnp.linalg.norm(xydir_cor)
		
		rotated_zdir = position_point_jit(0.0,0.0,jnp.pi,jnp.array([normed2]))[0]
		xycqp = jnp.array([0.0,xydir_cor[0],xydir_cor[1],xydir_cor[2]])
		xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
		xycrot_p = xycrot_qp[1:]
		
		rotated_xydir = position_point_jit(0.0,0.0,jnp.pi,jnp.array([xycrot_p]))[0]
		
		direc1 = jnp.cross(rotated_zdir,jnp.array([0.0,0.0,1.0]))
		ang1 = (1-1e-8)*jnp.dot(rotated_zdir,jnp.array([0.0,0.0,1.0]))
		
		rot1q = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
		rot1q /= jnp.linalg.norm(rot1q)
		rot1qi = qcong_jit(rot1q)
		
		xycqp = jnp.array([0.0,rotated_xydir[0],rotated_xydir[1],rotated_xydir[2]])
		xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
		xycrot_p = xycrot_qp[1:]

		flipped_data = flipped_data.at[2:5].set(rotated_zdir)
		flipped_data = flipped_data.at[5:7].set(xycrot_p[:2])
		flipped_data = flipped_data.at[7].set(-flipped_data[7])
		flipped_data = flipped_data.at[1].set(-flipped_data[1])
		def is_double():
			return self.calc_pot_jit(flipped_data[:num],self.mem_structure_om,self.mem_structure_im,1)
		def is_not_double():
			return self.calc_pot_jit(flipped_data[:num],self.mem_structure_im,self.mem_structure_om,1)
		final_flipped = jax.lax.cond(self.dbmem,is_double,is_not_double)
		def flip(final_data):
			return flipped_data
		def no_flip(final_data):
			return final_data
		final_data = jax.lax.cond(final_pot > final_flipped,flip,no_flip,final_data)
		final_pot_pos = jnp.zeros(num+1)
		final_pot_pos = final_pot_pos.at[:num].set(final_data[:num])
		final_pot_pos = final_pot_pos.at[num].set(self.calc_pot_jit(final_data[:num],self.mem_structure_im,self.mem_structure_om,0))
		return final_pot_pos
				
	#We turn minimise into a paralelised method
	@partial(jax.jit,static_argnums=2)
	def minimise_p(self,starting_pos,max_iter):
		return jax.pmap(self.minimise,static_broadcasted_argnums=1,in_axes=(0,None))(starting_pos,max_iter)
		

	# Calculating the hydrophobic core for each starting configuration	
	def get_starts(self,angs):
		starts = []
		angs_np = np.array(angs)
		for angs_ind in angs_np:
			start_zdir = position_point_jit(0,angs_ind[1],angs_ind[0],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,angs_ind[1],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			new_start_z = self.get_hydro_core_v3_p1(start_zdir,start_xydir)
			starts.append(new_start_z)
		return jnp.array(starts)	
		
	#A function that minimises on a set of different starting positions. This is important as there can be local minima
	def minimise_on_grid(self,grid_size,start_z,zdist,angs,max_iter):
		pos_grid = jnp.zeros((grid_size,9))
		def min_plot_fun_1(pos_grid,ind):
			start_zdir = position_point_jit(0,angs[ind][1],angs[ind][0],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,angs[ind][1],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			
			#Using a different starting insertion depth for peripheral proteins
			def is_peri():
				new_start_z = self.z_pot_no_range(100,start_zdir,start_xydir)
				
				return new_start_z
			def is_not_peri():
				def isdm():
					new_start_z = position_point_jit(0,angs[ind][1],angs[ind][0],jnp.array([start_z[0]]))[0,2]
					return new_start_z
				def isndm():
					new_start_z = start_z[ind,2]
					return new_start_z
				new_start_z = jax.lax.cond(self.dbmem,isdm,isndm)
				return new_start_z
			new_start_z = jax.lax.cond(self.peri,is_peri,is_not_peri)
			pos = jnp.concatenate([jnp.array([zdist,new_start_z]),start_zdir,start_xydir,jnp.zeros(1),jnp.zeros(1)])
			pos_grid = pos_grid.at[ind].set(pos)
			return pos_grid,ind
			
		no_runs = jnp.ceil(grid_size/no_cpu).astype(int)
		
		pos_grid,_ = jax.lax.scan(min_plot_fun_1,pos_grid,jnp.arange(grid_size))
		self.start_grid = pos_grid.copy()
		no_grid = pos_grid.shape[0]
		pos_grid =jnp.pad(pos_grid,((0,no_runs*no_cpu - no_grid),(0,0)),mode="edge")

		result_grid = jnp.zeros_like(pos_grid)
	
		print("Batch size:",no_cpu)
		print("Number of batches:",no_runs)
		times_taken = jnp.zeros(no_runs)
		#This non JAX for loop is needed beacuse pmap doesn't interact well with scan
		for i in range(no_runs):
			timers = time.time()
			print("Starting batch",str(i+1)+"/"+str(no_runs))
			result_grid = result_grid.at[i::no_runs].set(self.minimise_p(pos_grid[i::no_runs],max_iter))
			times_taken = times_taken.at[i].set((time.time()-timers))
			if(i > 5):
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[i-5:i+1]),dtype=int)
			elif(i>1):
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[1:i+1]),dtype=int)
			else:
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[:i+1]),dtype=int)
		self.result_grid=result_grid[:no_grid]
		jax.block_until_ready(result_grid)
		
	#A function to normalise positions to allow comparisons
	def normalise_pos(self,position):
		new_pos = jnp.zeros(4)
		new_pos = new_pos.at[:2].set(position[:2])
		zdir_test_a = position[2:5]
		zdir_test_a /= jnp.linalg.norm(zdir_test_a)
		new_pos = new_pos.at[3].set(jnp.arccos(jnp.dot(zdir_test_a,jnp.array([0.0,0.0,1.0]))))
		xydir_test = jnp.array([position[5],position[6],0])
		zdir_test = jnp.array([position[2],position[3],0])
		anga = 0.0
		def z_dir_zero(anga,xydir_test,zdir_test):
			return 0.0
		def z_dir_not_zero(anga,xydir_test,zdir_test):
			xydir_test /= jnp.linalg.norm(xydir_test)
			zdir_test /= jnp.linalg.norm(zdir_test)
			anga = jnp.arctan2(jnp.dot(jnp.cross(zdir_test,xydir_test),jnp.array([0.0,0.0,1.0])),jnp.dot(xydir_test,zdir_test))
			return anga
		anga = jax.lax.cond(jnp.linalg.norm(zdir_test)<1e-5,z_dir_zero,z_dir_not_zero,anga,xydir_test,zdir_test)

		new_pos = new_pos.at[2].set(anga)
		
		def norm_fun_1(new_pos):
			new_pos = new_pos.at[3].set(new_pos[3]-2*jnp.pi)
			return new_pos
		def norm_cond_1(new_pos):
			return new_pos[3] > jnp.pi
		
		def norm_fun_2(new_pos):
			new_pos = new_pos.at[3].set(new_pos[3]+2*jnp.pi)
			return new_pos
		def norm_cond_2(new_pos):
			return new_pos[3] < -jnp.pi
			
		new_pos = jax.lax.while_loop(norm_cond_1,norm_fun_1,new_pos)
		new_pos = jax.lax.while_loop(norm_cond_2,norm_fun_2,new_pos)
		
		def is_neg(new_pos):
			new_pos = new_pos.at[3].set(-new_pos[3])
			new_pos = new_pos.at[2].set(new_pos[2]+jnp.pi)
			return new_pos
		def is_pos(new_pos):
			return new_pos
		new_pos = jax.lax.cond(new_pos[3]< 0,is_neg,is_pos,new_pos)
		
		def norm_fun_3(ang):
			ang += jnp.pi*2
			return ang
		def norm_cond_3(ang):
			return ang < 0
			
		def norm_fun_4(ang):
			ang -= jnp.pi*2
			return ang
		def norm_cond_4(ang):
			return ang > 2*jnp.pi
			
		new_pos = new_pos.at[2].set(jax.lax.while_loop(norm_cond_3,norm_fun_3,new_pos[2]))
		new_pos = new_pos.at[2].set(jax.lax.while_loop(norm_cond_4,norm_fun_4,new_pos[2]))
		
		def in_water(new_pos):
			new_pos = jnp.zeros_like(new_pos)
			new_pos = new_pos.at[1].set(-300)
			return new_pos
		def not_in_water(new_pos):
			return new_pos
		return new_pos

	#A function for calculating which positions are equivilant. 
	#This will return all unique minima and there frequency
	def collect_minima_info(self,grid_size):
		pos_tracker = jnp.array([[0,0,1]],dtype="float64")
		results = jnp.reshape(self.result_grid,(grid_size,9))
		
		start_grid = jnp.reshape(self.start_grid,(grid_size,9))
		res_spos = jnp.zeros((grid_size,1,3))
		color_grid = jnp.zeros((grid_size,3),dtype="float64")
		pot_grid = self.result_grid[:,-1]
		sposs = jnp.zeros((grid_size,4))
		start_sposs = jnp.zeros((grid_size,4))
		def calc_spos(spos,ind):
			spos = spos.at[ind].set(self.normalise_pos(results[ind][:7]))
			return spos,ind
		sposs,_ = jax.lax.scan(calc_spos,sposs,jnp.arange(grid_size))
		
		
		osposs,_ = jax.lax.scan(calc_spos,sposs,jnp.arange(grid_size))
		
		def calc_spos(spos,ind):
			spos = spos.at[ind].set(self.normalise_pos(start_grid[ind][:7]))
			return spos,ind
		start_sposs,_ = jax.lax.scan(calc_spos,start_sposs,jnp.arange(grid_size))
		
		
		
		def calc_poses_fun_1(res_spos,ind):
			spos = sposs[ind]
			part_pos = position_point_jit(0,spos[2],spos[3],pos_tracker)
			res_spos = res_spos.at[ind].set(position_point_jit(0,-spos[2],0,part_pos))
			return res_spos,ind
			
		res_spos,_ = jax.lax.scan(calc_poses_fun_1,res_spos,jnp.arange(grid_size))
	
		#Collecting minima which are equivilant
		minima_types = jnp.zeros(grid_size+1,dtype = "int")
		dtol_a = jnp.pi/24.0
		dtol_z = 5
		dtol_p = 50
		ep = 1-1e-6
		def calc_no_poses_fun_1(minima_types,ind):
			def calc_no_poses_fun_2(ind_fix,ind):
				def is_same(ind_fix):
					ind_fix = ind_fix.at[1].set(0)
					ind_fix = ind_fix.at[2].set(ind+1)
					return ind_fix
				def is_not_same(ind_fix):
					return ind_fix
				distance = jnp.array([10.0,10.0,10.0])
				def go(distance):
					spos = sposs[ind]
					sposf = sposs[ind_fix[0]]
					distance = distance.at[0].set(jnp.arccos(ep*jnp.dot(res_spos[ind][0],res_spos[ind_fix[0]][0])))
					distance = distance.at[1].set(jnp.abs(spos[1]-sposf[1]))
					distance = distance.at[2].set(jnp.abs(results[ind][4]-results[ind_fix[0]][4]))
					return distance
				def stop(distance):
					return distance
				distance = jax.lax.cond(ind_fix[1] > 0.5,go,stop,distance)
				ind_fix = jax.lax.cond((distance[0] < dtol_a)*(distance[1] < dtol_z)*(distance[2] < dtol_p),is_same,is_not_same,ind_fix)
				return ind_fix,ind
			ind_fix,_ = jax.lax.scan(calc_no_poses_fun_2,jnp.array([ind,1,0],dtype = "int"),jnp.arange(grid_size))
			ind_type = ind_fix[2]
			def is_prev(ind_type):
				ind_type = minima_types[ind_type-1]
				return ind_type
			def is_not_prev(ind_type):
				return ind_type
			ind_type = jax.lax.cond(ind_type-1<ind,is_prev,is_not_prev,ind_type)
			def is_big(ind_type,minima_types):
				minima_types = minima_types.at[-1].set(minima_types[-1]+1)
				ind_type = minima_types[-1]
				return ind_type,minima_types
			def is_not_big(ind_type,minima_types):
				return ind_type,minima_types
			ind_type,minima_types = jax.lax.cond(ind_type > minima_types[-1],is_big,is_not_big,ind_type,minima_types)
			minima_types = minima_types.at[ind].set(ind_type)
			return minima_types,ind
			
		minima_types,_= jax.lax.scan(calc_no_poses_fun_1,minima_types,jnp.arange(grid_size))
		
		
				
		
		
		
		min_res_spos = jnp.zeros((minima_types[-1],8))
		def av_min_res_spos(min_res_spos,ind):
			spos = sposs[ind]
			correct_positon = position_point_jit(0,spos[2],spos[3],jnp.array([[0,0,1]]))
			correct_positon = position_point_jit(0,-spos[2],0,correct_positon)[0]
			min_res_spos = min_res_spos.at[minima_types[ind]-1,:3].set(min_res_spos[minima_types[ind]-1,:3]+correct_positon)
			min_res_spos = min_res_spos.at[minima_types[ind]-1,3].set(min_res_spos[minima_types[ind]-1,3]+spos[1])
			min_res_spos = min_res_spos.at[minima_types[ind]-1,4:].set(min_res_spos[minima_types[ind]-1,4:]+1)
			return min_res_spos,ind
		min_res_spos,_ = jax.lax.scan(av_min_res_spos,min_res_spos,np.arange(grid_size))
		min_res_spos = min_res_spos.at[:,:4].set(min_res_spos[:,:4]/min_res_spos[:,4:])
		
		
		prev_sposes = jnp.zeros(minima_types[-1]+1)+10
		prev_sposes = prev_sposes.at[minima_types[-1]].set(2.0)
		def fix_spos(prev_sposes,ind):
			spos = sposs[ind].copy()
			index = jnp.array(prev_sposes[-1],dtype=int)
			def first(prev_sposes):
				prev_sposes = prev_sposes.at[minima_types[ind]-1].set(spos[index])
				return prev_sposes
			def nfirst(prev_sposes):
				return prev_sposes
			prev_sposes = jax.lax.cond(prev_sposes[minima_types[ind]-1] > 9,first,nfirst,prev_sposes)
			def far(spos):
				def below(spos):
					spos = spos.at[index].set(spos[index]+2*jnp.pi)
					return spos
				def above(spos):
					spos = spos.at[index].set(spos[index]-2*jnp.pi)
					return spos
				return jax.lax.cond(prev_sposes[minima_types[ind]-1]>spos[index],below,above,spos)
				
			def nfar(spos):
				return spos		
				
			spos = jax.lax.cond(jnp.abs(prev_sposes[minima_types[ind]-1]-spos[index])>jnp.pi/2,far,nfar,spos)
			return prev_sposes,spos[index]
		_,sposs_ind2 = jax.lax.scan(fix_spos,prev_sposes,jnp.arange(grid_size))
		prev_sposes = prev_sposes.at[minima_types[-1]].set(3.0)
		_,sposs_ind3 = jax.lax.scan(fix_spos,prev_sposes,jnp.arange(grid_size))
		sposs = sposs.at[:,2].set(sposs_ind2)
		sposs = sposs.at[:,3].set(sposs_ind3)
		
				
		minima = jnp.zeros(((minima_types[-1])*5,4))
		def av_minima_fun_1(minima,ind):
			spos = sposs[ind]
			minima = minima.at[minima_types[ind]-1].set(minima[minima_types[ind]-1]+spos[:4])
			minima = minima.at[minima_types[ind]-1+minima_types[-1]].set(minima[minima_types[ind]-1+minima_types[-1]]+1)
			minima = minima.at[minima_types[ind]-1+2*minima_types[-1]].set(minima[minima_types[ind]-1+2*minima_types[-1]]+results[ind][8])
			minima = minima.at[minima_types[ind]-1+3*minima_types[-1]].set(minima[minima_types[ind]-1+3*minima_types[-1]]+spos[0]*jnp.cos(spos[3]))
			minima = minima.at[minima_types[ind]-1+4*minima_types[-1]].set(minima[minima_types[ind]-1+4*minima_types[-1]]+results[ind][7])
			return minima, ind
		minima,_ = jax.lax.scan(av_minima_fun_1,minima,jnp.arange(minima_types.shape[0]-1))
		minima = minima.at[:minima_types[-1]].set(minima[:minima_types[-1]]/minima[minima_types[-1]:2*minima_types[-1]])
		minima = minima.at[2*minima_types[-1]:3*minima_types[-1]].set(minima[2*minima_types[-1]:3*minima_types[-1]]/minima[minima_types[-1]:2*minima_types[-1]])
		minima = minima.at[3*minima_types[-1]:4*minima_types[-1]].set(minima[3*minima_types[-1]:4*minima_types[-1]]/minima[minima_types[-1]:2*minima_types[-1]])
		minima = minima.at[4*minima_types[-1]:].set(minima[4*minima_types[-1]:]/minima[minima_types[-1]:2*minima_types[-1]])
		minima_ind = jnp.zeros(minima_types[-1])
		if self.rank_sort == "h":
			minima_ind = jnp.argsort(minima[minima_types[-1]:2*minima_types[-1],0])[::-1]
		elif self.rank_sort == "p":
			minima_ind = jnp.argsort(minima[2*minima_types[-1]:3*minima_types[-1],0])
		minima = minima.at[:minima_types[-1]].set(minima[minima_ind])
		minima = minima.at[minima_types[-1]:2*minima_types[-1]].set(minima[minima_ind+minima_types[-1]])
		minima = minima.at[2*minima_types[-1]:3*minima_types[-1]].set(minima[minima_ind+2*minima_types[-1]])
		minima = minima.at[3*minima_types[-1]:4*minima_types[-1]].set(minima[minima_ind+3*minima_types[-1]])
		minima = minima.at[4*minima_types[-1]:].set(minima[minima_ind+4*minima_types[-1]])
		devs = jnp.zeros(minima_types[-1])
		
		#Getting deviations from rank 1
		def get_min_devs(devs,ind):
			dev1 = jnp.arccos(jnp.dot(min_res_spos[minima_ind[ind+1],:3],min_res_spos[minima_ind[0],:3]))
			dev2 = jnp.abs(min_res_spos[minima_ind[ind+1],3]-min_res_spos[minima_ind[0],3])/100
			devs = devs.at[minima_ind[ind+1]].set(dev1+dev2)
			return devs,ind
		devs,_ = jax.lax.scan(get_min_devs,devs,jnp.arange(minima_types[-1]-1))
		
		#Getting color associated with the deviations
		def color_grid_fun_2(color_grid,ind):
			spos = sposs[ind]
			colr = 1.0
			colg = 1-devs[minima_types[ind]-1]/3
			colb = 1-devs[minima_types[ind]-1]/3

			def gone(col):
				col = 1.0
				return col
			def lone(col):
				return col
			
			def lzero(col):
				col=0.0
				return col
			def gzero(col):
				return col
				
			colr = jax.lax.cond(colr > 1, gone,lone,colr)
			colg = jax.lax.cond(colg > 1, gone,lone,colg)
			colb = jax.lax.cond(colb > 1, gone,lone,colb)
	 
			colr = jax.lax.cond(colr < 0, lzero,gzero,colr)
			colg = jax.lax.cond(colg < 0, lzero,gzero,colg)
			colb = jax.lax.cond(colb < 0, lzero,gzero,colb)
			color_grid = color_grid.at[ind].set(jnp.array([colr,colg,colb]))
			return color_grid,ind
			
		color_grid,_ = jax.lax.scan(color_grid_fun_2,color_grid,jnp.arange(color_grid.shape[0]))

		self.minima = jnp.array(minima)
		
		self.minima_types = minima_types[:-1]
		self.minima_ind = minima_ind
		self.no_mins = minima_types[-1]
		self.new_memt_im = jnp.zeros(minima_types[-1])+self.memt_tails
		self.new_memt_om = jnp.zeros(minima_types[-1])+self.memt_tails_om
		
		self.re_rank_vals = jnp.zeros(minima_types[-1])
		self.re_rank_pots = jnp.zeros(minima_types[-1])
		self.re_rank_disses = jnp.zeros(minima_types[-1])		
		
		return jnp.array(color_grid),pot_grid
		
	def re_rank_minima(self,orient_dir):
		nminima_ind = jnp.zeros(self.no_mins)
		nminima_ind = jnp.argsort(self.re_rank_vals)[::-1]
		
		self.minima = self.minima.at[:self.no_mins].set(self.minima[nminima_ind])
		self.minima = self.minima.at[self.no_mins:2*self.no_mins].set(self.minima[nminima_ind+self.no_mins])
		self.minima = self.minima.at[2*self.no_mins:3*self.no_mins].set(self.minima[nminima_ind+2*self.no_mins])
		self.minima = self.minima.at[3*self.no_mins:4*self.no_mins].set(self.minima[nminima_ind+3*self.no_mins])
		self.minima = self.minima.at[4*self.no_mins:].set(self.minima[nminima_ind+4*self.no_mins])
		

		nminima_ind = jnp.array(nminima_ind)
		self.re_rank_pots = jnp.array(self.re_rank_pots)[nminima_ind]
		self.re_rank_vals = jnp.array(self.re_rank_vals)[nminima_ind]
		self.re_rank_disses = jnp.array(self.re_rank_disses)[nminima_ind]	
		
		self.minima_ind = jnp.array(self.minima_ind)[nminima_ind] #TODO?
		
		for i in range(self.no_mins):
			inder = np.where(nminima_ind == i)
			os.rename(orient_dir+"Rank_"+str(i+1)+"/Z_potential_curve.png",orient_dir+"Rank_"+str(i+1)+"/Z_potential_curve_"+str(inder[0][0]+1)+".png")
			os.rename(orient_dir+"Rank_"+str(i+1)+"/curv_potential_curve.png",orient_dir+"Rank_"+str(i+1)+"/curv_potential_curve_"+str(inder[0][0]+1)+".png")
			if(inder[0][0]+1 != i+1):
				shutil.move(orient_dir+"Rank_"+str(i+1)+"/Z_potential_curve_"+str(inder[0][0]+1)+".png",orient_dir+"Rank_"+str(inder[0][0]+1))
				shutil.move(orient_dir+"Rank_"+str(i+1)+"/curv_potential_curve_"+str(inder[0][0]+1)+".png",orient_dir+"Rank_"+str(inder[0][0]+1))
		for i in range(self.no_mins):	
			os.rename(orient_dir+"Rank_"+str(i+1)+"/Z_potential_curve_"+str(i+1)+".png",orient_dir+"Rank_"+str(i+1)+"/Z_potential_curve.png")
			os.rename(orient_dir+"Rank_"+str(i+1)+"/curv_potential_curve_"+str(i+1)+".png",orient_dir+"Rank_"+str(i+1)+"/curv_potential_curve.png")
	#This function optimises the membrane thickness (Should probably be JAXED)
	@partial(jax.jit,static_argnums=2)
	def optimise_mem_thickness(self,ind,fine,outer):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		min_hits = mins[no_mins:no_mins*2,0]
		min_hits = 100*min_hits/(np.sum(min_hits))
		min_pots = mins[no_mins*2:no_mins*3,0]
		min_zdist = jnp.abs(mins[no_mins*3:,0])
		min_curvs = mins[no_mins*4:0,0]
		numa = self.numa
		numb = self.numb
		
		zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir,jnp.array([min_curvs[ind]])])
		
		
		insd_grid = jnp.linspace(-5,5,fine)
		def is_outer3():
			return self.memt_tails_om
		def is_inner3():
			return self.memt_tails
		memt_start = jax.lax.cond(outer,is_outer3,is_inner3)
		memt_grid = jnp.linspace(memt_start-20,memt_start+20,fine)
		vals = jnp.zeros((fine,fine))
		def memt_fun1(vals,ind):
			ind_fix = ind
			def memt_fun2(vals,ind):
				
				def zero(change):
					return change,0.0
				def not_zero(change):
					def is_outer2(change):
						return -change/2
					def is_inner2(change):
						return change/2
					return change/2,jax.lax.cond(outer,is_outer2,is_inner2,change)
				chg1,chg2 = jax.lax.cond(position[0] < 1e-5,zero,not_zero,insd_grid[ind])
				position_test = position.at[1].set(position[1]+chg1)
				position_test = position_test.at[0].set(position[0]+chg2)
				
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				l_w = memt_grid[ind_fix]
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				def is_outer(mem_structure_im,mem_structure_om):
					return jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w]),mem_structure_im.copy()
				def is_inner(mem_structure_im,mem_structure_om):
					return mem_structure_om.copy(),jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				mem_structure_om,mem_structure_im = jax.lax.cond(outer,is_outer,is_inner,self.mem_structure_im,self.mem_structure_om)
				vals = vals.at[ind,ind_fix].set(self.calc_pot_jit(position_test,mem_structure_im,mem_structure_om,0))
				return vals, ind
			vals,_=jax.lax.scan(memt_fun2,vals,jnp.arange(fine))
			return vals,ind
		vals,_=jax.lax.scan(memt_fun1,vals,jnp.arange(fine))
		best = jnp.unravel_index(jnp.argmin(vals),(fine,fine))
		best_pos = insd_grid[best[0]]+position[1]
		best_memt = memt_grid[best[1]]
		return best_pos,best_memt
		
	#This function evaluates the optimal membrane thickness for all minima
	def optimise_memt_all(self):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_zdist = jnp.abs(mins[no_mins*3:,0])
		carry = jnp.zeros(5*no_mins)
		new_poses = jnp.zeros(no_mins)
		new_zdists = jnp.zeros(no_mins)
		min_poses = mins[:no_mins]
		
		def allmem_fun1(carry,ind):
			def zero(carry):
				bpos,bmem_im = self.optimise_mem_thickness(ind,10,False)
				carry = carry.at[ind].set(bpos)
				carry = carry.at[ind+no_mins*2].set(bmem_im)
				
				zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
				xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
				position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir])
				
				position_test = position.at[1].set(bpos)
				
				
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				l_w = bmem_im
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
					
				
				carry = carry.at[ind+no_mins*4].set(self.calc_pot_jit(position_test,mem_structure_im,self.mem_structure_om,0))
				return carry
			def not_zero(carry):
				bpos,bmem_im = self.optimise_mem_thickness(ind,10,False)
				bpos2,bmem_om = self.optimise_mem_thickness(ind,10,True)
				
				carry = carry.at[ind].set((bpos+bpos2)/2.0)
				carry = carry.at[ind+no_mins].set(min_zdist[ind]+(bpos-bpos2)/2.0)
				
				carry = carry.at[ind+no_mins*2].set(bmem_im)
				carry = carry.at[ind+no_mins*3].set(bmem_om)
				
				zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
				xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
				position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir])
				
				position_test = position.at[1].set((bpos+bpos2)/2.0)
				position_test = position_test.at[0].set(min_zdist[ind]+(bpos-bpos2)/2.0)
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				l_w = bmem_im
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				
				l_w = bmem_om
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_om = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				
				carry = carry.at[ind+no_mins*4].set(self.calc_pot_jit(position_test,mem_structure_im,mem_structure_om,0))
				
				return carry
			carry = jax.lax.cond(not self.dbmem,zero,not_zero,carry)
			return carry,ind
		carry,_ = jax.lax.scan(allmem_fun1,carry,jnp.arange(no_mins))
		self.new_memt_im = carry[no_mins*2:no_mins*3]
		self.new_memt_om = carry[no_mins*3:no_mins*4]
		new_poses = carry[:no_mins]
		new_zdists = carry[no_mins:no_mins*2]
		self.minima = self.minima.at[no_mins*3:no_mins*4,0].set(new_zdists)
		self.minima = self.minima.at[:no_mins,1].set(new_poses)
		
	#Calculates potential per bead for a paritcular final minima	
	def calc_pot_per_bead_ind(self,ind):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		min_curvs = mins[no_mins*4:,0]
		
		zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir,jnp.array([min_curvs[ind]])])
		
		heads = self.memt_heads
		h1_w = heads/6.0
		h2_w = heads/6.0
		h3_w = heads/6.0
				
		l_w = self.new_memt_im[ind]
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		
		l_w = self.new_memt_om[ind]
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		mem_structure_om = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		
		return self.calc_pot_per_bead(position,mem_structure_im,mem_structure_om)
	
	
	#Calculates the area of a convex hull
	def eval_chull(self,chull):
		chull0 = chull[0]
		mid = np.mean(chull0,axis=0)
		area = 0
		for i in range(chull0.shape[0]-1):
			lenny1 = np.linalg.norm(chull0[i]-mid)
			lenny2 = np.linalg.norm(chull0[i+1]-mid)
			area+= 0.5*lenny1*lenny2*np.sin(np.arccos(np.dot(chull0[i]-mid,chull0[i+1]-mid)/(lenny2*lenny1)))
		return area
	
	#A function for calculating the cross sectional area of a protein in the PG layer
	def area_calc_v2np(self,points):
		cen = np.mean(points,axis=0)
		disters = np.linalg.norm(points-cen,axis=1)
		dirs =  np.array([(points[ind]-cen)/disters[ind] for ind in range(points.shape[0])])
		f_ind = np.argmax(disters)
		f_point = points[f_ind]
		convex_hull = np.zeros((points.shape[0]+1),dtype=int)
		convex_hull[0] = f_ind
		convex_hull[-1] = 1
		
		
		while ((convex_hull[convex_hull[-1]-1] != convex_hull[0])+(convex_hull[-1]==1))*(convex_hull.shape[0]>10):
			disters = np.linalg.norm(points-points[convex_hull[convex_hull[-1]-1]],axis=1)
			dirs2 = np.array([(points[ind]-points[convex_hull[convex_hull[-1]-1]])/disters[ind] for ind in range(points.shape[0])])
			direr = dirs[convex_hull[convex_hull[-1]-1]]
			all_dots = np.zeros(dirs2.shape[0])
			for ind in range(dirs2.shape[0]):
				ang1 = np.arctan2(direr[1],direr[0])
				rot_mat = np.array([[np.cos(-ang1),-np.sin(-ang1)],[np.sin(-ang1),np.cos(-ang1)]])
				dt2 = np.dot(rot_mat,dirs2[ind][:2])
				ang2 = np.arctan2(dt2[1],dt2[0])
				all_dots[ind] = ang2
			all_dots = np.where(all_dots<0,all_dots+2*np.pi,all_dots)
			first = np.nanargmin(all_dots)
			convex_hull[convex_hull[-1]] = first
			convex_hull[-1] = convex_hull[-1]+1
		return convex_hull
		
	#Uses the above to create a graph
	def get_area_graph(self,min_ind,nums):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		ranger = jnp.abs(min_poses[min_ind][0]*np.cos(min_poses[min_ind][3]))
		xs = np.linspace(-ranger-self.mem_structure[0]+self.pg_thickness/2,ranger+self.mem_structure[0]-self.pg_thickness/2,nums)
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		test_pos = jnp.concatenate([jnp.array([min_poses[min_ind][0],min_poses[min_ind][1]]),start_zdir,start_xydir])
		all_areas = np.zeros(nums)
		
		zdist_temp = test_pos[0]
		in_depth = test_pos[1]
		ang1 = test_pos[2:5]
		ang2 = test_pos[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = np.array(position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses))
		
		
		for i,x in enumerate(xs):
			testing = tester_poses[tester_poses[:,2]>x-self.pg_thickness/2]
			testing = testing[testing[:,2]<x+self.pg_thickness/2]
			testing[:,2] = 0
			
			chull = self.area_calc_v2np(np.array(testing))
			hull_pts = chull[:chull[-1]]
			
			all_areas[i] = self.eval_chull(testing[[hull_pts]])
		return all_areas,xs
		
	#Getting the area at the minima
	def get_area_pos(self,min_ind,pos):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		test_pos = jnp.concatenate([jnp.array([min_poses[min_ind][0],min_poses[min_ind][1]]),start_zdir,start_xydir])		
		zdist_temp = test_pos[0]
		in_depth = test_pos[1]
		ang1 = test_pos[2:5]
		ang2 = test_pos[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = np.array(position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses))
		
		testing = tester_poses[tester_poses[:,2]>pos-self.pg_thickness/2]
		testing = testing[testing[:,2]<pos+self.pg_thickness/2]
		testing[:,2] = 0
		
		chull = self.area_calc_v2np(np.array(testing))
		hull_pts = chull[:chull[-1]]
	
		return self.eval_chull(testing[[hull_pts]])
	
	#Functionto get all the PG layer position across different minima
	def get_all_pg_pos(self,orient_dir,pg_guess):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_zdist = jnp.abs(mins[no_mins*3:,0])
		min_poses = mins[:no_mins]
		self.pg_poses = np.zeros(no_mins)
		no_areas = 25
		self.all_areas = np.zeros((no_mins,2,no_areas))
		def pg_fun(carry,ind):
			heads = self.memt_heads
			h1_w = heads/6.0
			h2_w = heads/6.0
			h3_w = heads/6.0
			
			l_w = self.new_memt_im[ind]
			meml = -l_w/2.0 -h1_w -h2_w-h3_w
			mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
			
			l_w = self.new_memt_om[ind]
			meml = -l_w/2.0 -h1_w -h2_w-h3_w
			mem_structure_om = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
			
			pg_pos,pg_xs = self.get_pg_pos(400,ind,mem_structure_im,mem_structure_om,pg_guess)
			return carry,jnp.array([pg_pos,pg_xs])
		_,self.pg_poses = jax.lax.scan(pg_fun,0,jnp.arange(no_mins))
		
		for ind in range(no_mins):
			areas,area_xs = self.get_area_graph(ind,no_areas)
			self.all_areas[ind] = np.array([areas,area_xs])
		
	#This function is used approximate the minima depth. It looks only at the potential when insertion depth is changed.
	def approx_minima_depth(self,min_ind,rank_dir,orient_dir):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		curv = mins[no_mins*4+min_ind,0]
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]	
		in_depth = min_poses[min_ind][1]
		pot_graph,zs,in_depth_ind = self.z_pot_graph(150,150,start_zdir,start_xydir,in_depth,curv)

		if not self.dbmem:
			pots_cs,cs= self.c_pot_plot(150,start_zdir,start_xydir,in_depth)
			
			
			plt.plot(cs/10,pots_cs)
			plt.ylabel("Potential energy")
			plt.xlabel("Z")
			plt.title("Potential energy against curvature")
			plt.tight_layout()
			plt.savefig(rank_dir+"curv_potential_curve.png")
			plt.clf()
		
		plt.plot(zs,pot_graph)
		plt.ylabel("Potential energy")
		plt.xlabel("Z")
		plt.title("Potential energy against z position relative to centre of membrane")
		plt.tight_layout()
		plt.savefig(rank_dir+"Z_potential_curve.png")
		plt.clf()
		
		if(in_depth_ind > 148):
			in_depth_ind = 148
		elif(in_depth_ind < 1):
			in_depth_ind = 1
		minn = jnp.max(pot_graph[:in_depth_ind])-pot_graph[in_depth_ind]
		maxx = jnp.max(pot_graph[in_depth_ind:])-pot_graph[in_depth_ind]
		diss = min(minn,maxx)	
		return diss,pot_graph[in_depth_ind]
		
	#A function that runs the above on all minima
	def approx_minima_depth_all(self,scaler,orient_dir):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//5)
		disses = np.zeros(no_mins)
		pots = np.zeros(no_mins)
		for ind in range(no_mins):
			if(not os.path.exists(orient_dir+"Rank_"+str(ind+1)+"/")):
				os.mkdir(orient_dir+"Rank_"+str(ind+1)+"/")
			rank_dir = orient_dir+"Rank_"+str(ind+1)+"/"
			diss,pot = self.approx_minima_depth(ind,rank_dir,orient_dir)
			disses[ind] = diss
			pots[ind] = pot
		self.re_rank_pots = pots
		self.re_rank_disses = disses
		max_pot = np.max(pots)
		pots = pots - max_pot
		disses = (disses*scaler +max_pot)
		self.re_rank_vals = disses*disses+pots*pots
		
		
	#Builds a CG system using insane
	def build_oriented(self,orient_dir,build_args):
		mins = np.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		min_zdist = mins[no_mins*3:,0]
		min_curvs = mins[no_mins*4:,0]/10
		no_build = min(self.build_no,no_mins)
		for i in range(no_build):
			rank_dir = orient_dir+"Rank_"+str(i+1)+"/"
			zdist = min_zdist[i]/10
			
			start_zdir = position_point_jit(0,min_poses[i][2],min_poses[i][3],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,min_poses[i][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			posi = jnp.zeros(7)
			posi = posi.at[2:5].set(start_zdir)
			posi = posi.at[5:7].set(start_xydir)
			x_min,x_max,y_min,y_max,z_min,z_max = self.get_extent(posi)
			x_range = (80+x_max-x_min)/10.0
			y_range = (80+y_max-y_min)/10.0
			z_range = (100+z_max-z_min)/10.0
			curv_str = " -curv "+str(10*np.abs(min_curvs[i]))+",0.1,"+str(np.sign(min_curvs[i]-1e-9))
			print("Building rank "+str(i+1)+" with Insane4MemPrO:")
			cg_sys_dir = rank_dir+"CG_System_rank_"+str(i+1)+"/"
			run_str = "python "+PATH_TO_INSANE+" "+build_args.strip()+" -o "+cg_sys_dir+"CG-system.gro -p "+cg_sys_dir+"topol.top -f "+cg_sys_dir+"protein-cg.pdb "+"-x "+str(x_range)+" -y "+str(y_range)+" -z "+str(z_range)+curv_str
			if(zdist > 1e-6):
				run_str += " -ps "+str(zdist)
			if(self.pg_layer_pred):
				ps = self.pg_poses[i][0]
				xs_pg = self.pg_poses[i][1]
				pg_pos = -xs_pg[jnp.argmin(ps)]
				run_str += " -pgl_z "+str(pg_pos/10.0)+" -o_f "+cg_sys_dir
				#print()
			err_val = os.system(run_str)
			if(err_val != 0):
				print("WARNING: There was an error when trying to build the system. Check -bd_args are correct.")
	#Writes a pdb using a given file as a template and replacing positions
	def write_oriented(self,temp_fn,orient_dir,run_cmd):
		lfile = open(temp_fn,"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		

		mins = np.array(self.minima)
		no_mins = (mins.shape[0]//5)
		min_poses = mins[:no_mins]
		min_hits = mins[no_mins:no_mins*2,0]
		min_hits = 100*min_hits/(np.sum(min_hits))
		min_pots = mins[no_mins*2:no_mins*3,0]
		min_zdist = np.abs(mins[no_mins*3:no_mins*4,0])
		min_curvs = mins[no_mins*4:,0]/10

		
		
		npmem_im = np.array(self.new_memt_im)
		npmem_om = np.array(self.new_memt_om)

		ranks = open(os.path.join(orient_dir,"orientation.txt"),"w")
		all_orients = open(os.path.join(orient_dir,"orientations.pdb"),"w")
		ranks.write("Generated by command: "+run_cmd+"\n")
		ranks.write("Rank \t Relative potential \t % hits \t Re-rank pots \t Re-rank minima depth \t Re-rank values\n")# \t force max + \t force max -\n")
		for i in range(min_poses.shape[0]):
			if(not os.path.exists(orient_dir+"Rank_"+str(i+1)+"/")):
				os.mkdir(orient_dir+"Rank_"+str(i+1)+"/")
			rank_dir = orient_dir+"Rank_"+str(i+1)+"/"
			infos = open(os.path.join(rank_dir,"info_rank_"+str(i+1)+".txt"),"w")
			infos.write("Iter-Membrane distance (for -dm only): "+str(form(2*min_zdist[i]))+" A \n")
			infos.write("Potential Energy (rel): "+str(form(min_pots[i]))+"\n")
			infos.write("Global curvature (-c): "+str(form(min_curvs[i]))+" A^-1 \n")
			infos.write("(Inner) Membrane Thickness: "+str(form(10+self.new_memt_im[i]))+" A \n")
			infos.write("Outer Membrane Thickness (for -dm only): "+str(form(10+self.new_memt_om[i]))+" A \n")
			infos.write("Re-rank Pots (for not -dm only): "+str(form(self.re_rank_pots[i]))+"\n")
			infos.write("Re-rank minima depth (for not -dm only): "+str(form(self.re_rank_disses[i]))+"\n")
			infos.write("Re-rank final values (for not -dm only): "+str(form(self.re_rank_vals[i]))+"\n")
			
			start_zdir = position_point_jit(0,min_poses[i][2],min_poses[i][3],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,min_poses[i][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			
			orient_poses = np.array(position_point_jit(min_poses[i][1],min_poses[i][2],min_poses[i][3],self.all_poses))
			ranks.write(str(i+1)+"\t "+str(form(min_pots[i]))+"\t"+str(form(min_hits[i]))+"\t"+str(form(self.re_rank_pots[i]))+"\t"+str(form(self.re_rank_disses[i]))+"\t"+str(form(self.re_rank_vals[i]))+"\n")#+"\t"+str(pm)+"\t"+str(mm)+"\n")
			new_file = open(os.path.join(rank_dir,"oriented_rank_"+str(i+1)+".pdb"),"w")
			if(self.build_no > i):
				cg_sys_dir = rank_dir+"CG_System_rank_"+str(i+1)
				if(not os.path.exists(cg_sys_dir)):
					os.mkdir(cg_sys_dir)
				new_file_b = open(os.path.join(cg_sys_dir,"protein-cg.pdb"),"w")
			count = 0
			count2 = 0
			neww = False
			prev_num = -1
			strr = ""
			strc = ""
			count_pb =-1
			pot_pb = np.zeros(self.surface_poses.shape[0])
			
			if(not self.curva and self.write_bfac):
				pot_pb = self.calc_pot_per_bead_ind(i)
				pot_pb = np.array(pot_pb)
			for c in content:
				if len(c) > 54 and "[" not in c and c.startswith("ATOM"):
					zpos = c[46:54]
					ypos = c[38:46]
					xpos = c[30:38]
					res = c[17:20].strip()
					atom_num = int(c[22:26].strip())
					b_val = float(c[60:66].strip())
					pos = np.array([float(xpos.strip()),float(ypos.strip()),float(zpos.strip())])
					if(not np.any(np.isnan(pos))):
						xp = np.format_float_positional(orient_poses[count][0],precision=3)
						yp = np.format_float_positional(orient_poses[count][1],precision=3)
						zp = np.format_float_positional(orient_poses[count][2],precision=3)
						xp += "0"*(3-len((xp.split(".")[1])))
						yp += "0"*(3-len((yp.split(".")[1])))
						zp += "0"*(3-len((zp.split(".")[1])))
						if(atom_num != prev_num):
							if(neww):
								strr += three2one(res)
								strc += "T"
							else:
								strr += three2one(res)
								strc += "N"
							neww = False
						bbp = np.format_float_positional(b_val,precision=3)
						if(self.dbmem):
							if((npmem_im[i]/2 > orient_poses[count][2]+min_zdist[i] >-npmem_im[i]/2) or (npmem_om[i]/2 > orient_poses[count][2]-min_zdist[i] >-npmem_om[i]/2)):
								neww = True
						else:
							if(npmem_im[i]/2 > orient_poses[count][2] >-npmem_im[i]/2):
								neww = True
						npsurf = np.array(self.surface)
						if(not self.curva and self.write_bfac):
							if(npsurf[self.map_to_beads[count]] == 1):
								count_pb += 1
								no_bead = int(np.sum(npsurf[:self.map_to_beads[count]]))
								bbp = np.format_float_positional(pot_pb[no_bead],precision=3)
							else:
								bbp = np.format_float_positional(0,precision=3)
						
						bbp += "0"*(3-len((bbp.split(".")[1])))
						new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:60]+(" "*(8-len(bbp)))+bbp+c[66:]+"\n"						
						new_file.write(new_c)
						all_orients.write(new_c)
						if(self.build_no > i):
							new_file_b.write(new_c)
						count += 1
					prev_num = atom_num
			infos.write("Transmembrane residues (T):\n")
			infos.write(strr+"\n")
			infos.write(strc+"\n")
			grid_num = 50
			xs = jnp.linspace(-100,100,grid_num)
			ys = jnp.linspace(-100,100,grid_num)
			grid = []
			grid2 = []
			grid_norm = []
			
			xext = max(np.max(orient_poses[:,0]),-np.min(orient_poses[:,0]))
			yext = max(np.max(orient_poses[:,1]),-np.min(orient_poses[:,1]))
			big_ext = max(xext,yext)+10
			
			rader = np.abs(1/min_curvs[i])
			if(big_ext < rader):
				anger = np.arcsin(big_ext/rader)
			else:
				anger = np.pi
			
			if(min_curvs[i] > 0):
				xyz,norms = SphereGridN(0.4,anger,rader)
				xyz[:,2] *= -1
			elif(min_curvs[i] < 0):
				xyz,norms = SphereGridN(0.4,anger,rader)
				norms[:,:2] *= -1
			else:
				xyz,norms = DiskGrid(0.4,big_ext)
			count = 0
			xs = np.array(xs)
			ys = np.array(ys)
			for zind in range(2):
				zz = (2*zind)-1
				mem_totals = [self.memt_heads_om+self.new_memt_om[i],self.memt_heads+self.new_memt_im[i]]
				mem_tails = [self.new_memt_om[i],self.new_memt_im[i]]
				if(zz == 1 or abs(min_zdist[i]) > 1e-5):
					for xi in range(xyz.shape[0]):
						pos = xyz[xi]
						norma = norms[xi]
						zs = [mem_totals[zind]/2.0+zz*min_zdist[i],mem_tails[zind]/2.0+zz*min_zdist[i],-mem_tails[zind]/2.0+zz*min_zdist[i],-mem_totals[zind]/2.0+zz*min_zdist[i]]
						for xk in zs:
							count += 1
							count_str = (6-len(str(count)))*" "+str(count)
							c = "ATOM "+count_str+" BB   DUM Z   1	   0.000   0.000  15.000  1.00  0.00" 
							xp = np.format_float_positional(pos[0]+xk*norma[0],precision=3)
							yp = np.format_float_positional(pos[1]+xk*norma[1],precision=3)
							zp = np.format_float_positional(pos[2]+xk*norma[2],precision=3)
							xp += "0"*(3-len((xp.split(".")[1])))
							yp += "0"*(3-len((yp.split(".")[1])))
							zp += "0"*(3-len((zp.split(".")[1])))
							new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:]+"\n"	
							new_file.write(new_c)
							all_orients.write(new_c)
			if(abs(min_zdist[i])>1e-5 and self.pg_layer_pred): 
				ps = self.pg_poses[i][0]
				xs_pg = self.pg_poses[i][1]
				plt.plot(-xs_pg[jnp.abs(ps)<3000],ps[jnp.abs(ps)<3000])
				bot,top = plt.ylim()
				plt.title("Potential energy of PG layer at z")
				plt.ylabel("Potential energy(rel)")
				plt.xlabel("z")
				plt.tight_layout()
				plt.savefig(rank_dir+"PG_potential_curve.png")
				plt.clf()
				
				areas = self.all_areas[i][0]
				xs_areas = self.all_areas[i][1]
				plt.plot(xs_areas,areas)
				bot,top = plt.ylim()
				plt.title("Cross-sectional area at z")
				plt.ylabel("Area (A^2)")
				plt.xlabel("z")
				plt.tight_layout()
				plt.savefig(rank_dir+"cross-section_area_curve.png")
				plt.clf()
				
				
				pg_pos = -xs_pg[jnp.argmin(ps)]
				area = self.get_area_pos(i,pg_pos)
				infos.write("PG layer position (for -pg only): "+str(form(pg_pos))+" A \n")
				infos.write("Cross-sectional area in PG layer (for -pg only): "+str(form(area))+" A^2 \n")
				xyz,norms = DiskGrid(0.4,big_ext)
				for pos in xyz:
					count += 1
					count_str = (6-len(str(count)))*" "+str(count)
					c = "ATOM "+count_str+" BB   PGL Z   1	   0.000   0.000  15.000  1.00  0.00" 
					xp = np.format_float_positional(pos[0],precision=3)
					yp = np.format_float_positional(pos[1],precision=3)
					zpa = np.format_float_positional(pg_pos+self.pg_thickness/2,precision=3)
					zpb = np.format_float_positional(pg_pos-self.pg_thickness/2,precision=3)
					xp += "0"*(3-len((xp.split(".")[1])))
					yp += "0"*(3-len((yp.split(".")[1])))
					zpa += "0"*(3-len((zpa.split(".")[1])))
					zpb += "0"*(3-len((zpb.split(".")[1])))
					new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zpa))) +zpa+c[54:]+"\n"	
					new_file.write(new_c)
					all_orients.write(new_c)
					new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zpb))) +zpb+c[54:]+"\n"	
					new_file.write(new_c)
					all_orients.write(new_c)
			new_file.close()
			all_orients.write("TER\nEND\n")
			infos.close()
		ranks.close()
		all_orients.close()

	

	#A function that makes grids for the output graphs
	def make_graph_grids(self,angs,grid_size):
		new_minima_types = jnp.zeros_like(self.minima_types)
		no_mins = self.minima_ind_c.shape[0]
		no_mins2 = self.minima_ind.shape[0]
		
		minima_ind_inv = jnp.array([jnp.arange(no_mins2)[self.minima_ind==i][0] for i in range(no_mins2)])
		minima_ind_c_inv = jnp.array([jnp.arange(no_mins)[self.minima_ind_c==i][0] for i in range(no_mins)])
		
		def new_minima_types_fun(new_minima_types,ind):
			new_minima_types = new_minima_types.at[ind].set(minima_ind_c_inv[self.minima_types_c[minima_ind_inv[self.minima_types[ind]-1]]-1])
			return new_minima_types,ind
		
		new_minima_types,_ = jax.lax.scan(new_minima_types_fun,new_minima_types,jnp.arange(grid_size))
		pot_grid = jnp.zeros(grid_size)
		def set_pot_grid(pot_grid,ind):
			pot_grid = pot_grid.at[ind].set(self.minima_c[no_mins*2+new_minima_types[ind],0])
			return pot_grid,ind
		pot_grid,_ = jax.lax.scan(set_pot_grid,pot_grid,jnp.arange(grid_size))
		devs = jnp.zeros(no_mins)
		ep = 1-1e-6
		def get_min_devs(devs,ind):
			sposind_p = position_point_jit(0,self.minima_c[ind+1][2],self.minima_c[ind+1][3],jnp.array([[0,0,1]]))
			sposind = position_point_jit(0,-self.minima_c[ind+1][2],0,sposind_p)[0]
			sposo_p = position_point_jit(0,self.minima_c[0][2],self.minima_c[0][3],jnp.array([[0,0,1]]))
			sposo = position_point_jit(0,-self.minima_c[0][2],0,sposo_p)[0]
			dev1 = jnp.arccos(jnp.dot(sposind,sposo)*ep)
			dev2 = jnp.abs(self.minima_c[ind+1][0]-self.minima_c[0][0])/100
			devs = devs.at[ind+1].set(dev1+dev2)
			return devs,ind
		devs,_ = jax.lax.scan(get_min_devs,devs,jnp.arange(no_mins))
		
		color_grid = jnp.zeros((grid_size,3),dtype="float64")
		def color_grid_fun_2(color_grid,ind):
			colr = 1.0
			colg = 1-devs[new_minima_types[ind]]/3
			colb = 1-devs[new_minima_types[ind]]/3

			def gone(col):
				col = 1.0
				return col
			def lone(col):
				return col
			
			def lzero(col):
				col=0.0
				return col
			def gzero(col):
				return col
				
			colr = jax.lax.cond(colr > 1, gone,lone,colr)
			colg = jax.lax.cond(colg > 1, gone,lone,colg)
			colb = jax.lax.cond(colb > 1, gone,lone,colb)
	 
			colr = jax.lax.cond(colr < 0, lzero,gzero,colr)
			colg = jax.lax.cond(colg < 0, lzero,gzero,colg)
			colb = jax.lax.cond(colb < 0, lzero,gzero,colb)
			color_grid = color_grid.at[ind].set(jnp.array([colr,colg,colb]))
			return color_grid,ind
			
		color_grid,_ = jax.lax.scan(color_grid_fun_2,color_grid,jnp.arange(color_grid.shape[0]))
		
		return pot_grid,color_grid
	
	
#Registering the class as a pytree node for JAX
tree_util.register_pytree_node(MemBrain,MemBrain._tree_flatten,MemBrain._tree_unflatten)
tree_util.register_pytree_node(PDB_helper,PDB_helper._tree_flatten,PDB_helper._tree_unflatten)

#A helper function the writes and arbritrary PDB file
def write_point(points,fn,orient_dir):
	new_file = open(os.path.join(orient_dir,fn),"w")
	count = 0
	for i in points:
		count += 1
		count_str = (6-len(str(count)))*" "+str(count)
		c = "ATOM "+count_str+" BB   DUM	 1	   0.000   0.000  15.000  1.00  0.00" 
		xp = np.format_float_positional(i[0],precision=3)
		yp = np.format_float_positional(i[1],precision=3)
		zp = np.format_float_positional(i[2],precision=3)
		xp += "0"*(3-len((xp.split(".")[1])))
		yp += "0"*(3-len((yp.split(".")[1])))
		zp += "0"*(3-len((zp.split(".")[1])))
		new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:]+"\n"	
		new_file.write(new_c)
	new_file.close()
	
	
#A sigmoid function
def sigmoid(x,grad):
	ret_val = -x*grad
	def overflow_pos(ret_val):
		ret_val = 100.0
		return ret_val
	def overflow_neg(ret_val):
		ret_val = -100.0
		return ret_val
	def not_overflow(ret_val):
		return ret_val
	
	ret_val = jax.lax.cond(-x*grad>100,overflow_pos,not_overflow,ret_val)
	ret_val = jax.lax.cond(-x*grad<-100,overflow_neg,not_overflow,ret_val)
	return 1.0/(1.0+jnp.exp(ret_val))
	
sj = jax.jit(sigmoid)


#Setting up quaternions for rotations later
def qmul(q1,q2):
	part1 = q1[0]*q2
	part2 = jnp.array([-q1[1]*q2[1],q1[1]*q2[0],-q1[1]*q2[3],q1[1]*q2[2]])
	part3 = jnp.array([-q1[2]*q2[2],q1[2]*q2[3],q1[2]*q2[0],-q1[2]*q2[1]])
	part4 = jnp.array([-q1[3]*q2[3],-q1[3]*q2[2],q1[3]*q2[1],q1[3]*q2[0]])
	return part1+part2+part3+part4
qmul_jit = jax.jit(qmul)	


def qcong(q1):
	q1 = q1.at[1:].set(-1*q1[1:])
	return q1
qcong_jit = jax.jit(qcong)

#function for calculating potential between two charged sheets
@partial(jax.jit,static_argnums=(1,2,3))
def potential_between_sheets(rdist,grid_size,extent,cc):
	xs = jnp.linspace(-extent,extent,grid_size)
	tot_pot = 0
	def pbs1(tot_pot,ind):
		ind_fix1=ind
		def pbs2(tot_pot,ind):
			point_a = jnp.array([xs[ind],xs[ind_fix1],0.0])
			ind_fix2 = ind
			def pbs3(tot_pot,ind):
				ind_fix3 = ind
				def pbs4(tot_pot,ind):
					point_b = jnp.array([xs[ind_fix3],xs[ind],rdist])
					tot_pot += (extent/grid_size)*(extent/grid_size)*(extent/grid_size)*(extent/grid_size)/jnp.linalg.norm(point_a-point_b)
					return tot_pot,ind
				tot_pot,_=jax.lax.scan(pbs4,tot_pot,jnp.arange(grid_size))
				return tot_pot,ind
			tot_pot,_=jax.lax.scan(pbs3,tot_pot,jnp.arange(grid_size))
			return tot_pot,ind
		tot_pot,_=jax.lax.scan(pbs2,tot_pot,jnp.arange(grid_size))
		return tot_pot,ind
	tot_pot,_=jax.lax.scan(pbs1,tot_pot,jnp.arange(grid_size))
	return tot_pot*cc
	
#A function for positioning a set of points
@jax.jit
def position_point_jit(in_depth,ang1,ang2,poses):
	pos_num = poses.shape[0]
	new_num = pad_num(poses)

	direc1 = jnp.array([0,0,1],dtype="float64")
	direc2 = jnp.array([1,0,0],dtype="float64")
	rot1q = jnp.array([jnp.cos(ang1/2),direc1[0]*jnp.sin(ang1/2),direc1[1]*jnp.sin(ang1/2),direc1[2]*jnp.sin(ang1/2)])
	rot2q = jnp.array([jnp.cos(ang2/2),direc2[0]*jnp.sin(ang2/2),direc2[1]*jnp.sin(ang2/2),direc2[2]*jnp.sin(ang2/2)])
	rot1qi = qcong_jit(rot1q)
	rot2qi = qcong_jit(rot2q)
	rotated_poses = poses.copy()
	@jax.vmap
	def postion_fun_1(ind):
		p_to_rot = ind
		qp = jnp.array([0,p_to_rot[0],p_to_rot[1],p_to_rot[2]])
		rot_qp1 = qmul_jit(rot1q,qmul_jit(qp,rot1qi))
		rot_qp = qmul_jit(rot2q,qmul_jit(rot_qp1,rot2qi))
		rot_p = rot_qp[1:]
		return rot_p
	rotated_poses = postion_fun_1(rotated_poses)
	rotated_poses = rotated_poses.at[:,2].set(rotated_poses[:,2]-in_depth)
	
	return rotated_poses



#A function for positioning a set of points (in a smoother manner as this is better for minimisation)
def position_pointv2(in_depth,zdir,xydir,poses):
	ep = 1-1e-8
	zdir /= jnp.linalg.norm(zdir)
	
	
	direc1 = jnp.cross(zdir,jnp.array([0.0,0.0,1.0]))
	ang1 = ep*jnp.dot(zdir,jnp.array([0.0,0.0,1.0]))
	
	rot1qi = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
	rot1qi /= jnp.linalg.norm(rot1qi)
	rot1q = qcong_jit(rot1qi)
	
	
	
	xydir_cor = jnp.array([xydir[0],xydir[1],0.0])
	xydir_cor /= jnp.linalg.norm(xydir_cor)
	
	xycqp = jnp.array([0.0,xydir_cor[0],xydir_cor[1],xydir_cor[2]])
	xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
	xycrot_p = xycrot_qp[1:]
	
	xyqp = jnp.array([0.0,0.0,-1.0,0.0])
	xyrot_qp = qmul_jit(rot1q,qmul_jit(xyqp,rot1qi))
	xyrot_p = xyrot_qp[1:]
	
	direc2 = jnp.cross(xyrot_p,xycrot_p)
	ang2 = ep*jnp.dot(xyrot_p,xycrot_p)
	
	rot2q = jnp.array([ang2+1,direc2[0],direc2[1],direc2[2]])
	rot2q /= jnp.linalg.norm(rot2q)
	rot2qi = qcong_jit(rot2q)
	
	
	rotated_poses = poses.copy()
	
	@jax.vmap
	def postion_fun_1(ind):
		p_to_rot = ind
		qp = jnp.array([0,p_to_rot[0],p_to_rot[1],p_to_rot[2]])
		rot_qp1 = qmul_jit(rot1q,qmul_jit(qp,rot1qi))
		rot_qp = qmul_jit(rot2q,qmul_jit(rot_qp1,rot2qi))
		rot_p = rot_qp[1:]
		return rot_p
	rotated_poses = postion_fun_1(rotated_poses)#jax.lax.scan(postion_fun_1,0,jnp.arange(poses.shape[0]))
	rotated_poses = rotated_poses.at[:,2].set(rotated_poses[:,2]-in_depth)
	
	return rotated_poses
position_pointv2_jit = jax.jit(position_pointv2)

#A function that returns an array of points on a sphere using a fibbonacci spiral lattice
def create_sph_grid(bsize):
	sgrid = jnp.zeros((bsize,2))
	gr = (1+jnp.sqrt(5))/2
	def ball_fun_3(sgrid,ind):
		phi = jnp.arccos(1-2*(ind+0.5)/(bsize*2))
		theta = jnp.pi*(ind+0.5)*(gr)*2
		def norm_fun_1(theta):
			theta = theta - 2*jnp.pi
			return theta
		def norm_cond_1(theta):
			return theta > jnp.pi*2
		
		def norm_fun_2(theta):
			theta = theta + 2*jnp.pi
			return theta
		def norm_cond_2(theta):
			return theta < 0
			
		theta = jax.lax.while_loop(norm_cond_1,norm_fun_1,theta)
		theta = jax.lax.while_loop(norm_cond_2,norm_fun_2,theta)
		sgrid = sgrid.at[ind].set(jnp.array([phi,theta]))
		return sgrid, ind
		
	sgrid,_ = jax.lax.scan(ball_fun_3,sgrid,np.arange(bsize))
	return sgrid
	
def create_sph_grid_charg(bsize,end_ang,orad):
	bsize_true = jnp.array(end_ang,dtype=int)
	
	
	sgrid = jnp.zeros((bsize,3))
	gr = (1+jnp.sqrt(5))/2
	def ball_fun_3(sgrid,ind):
		phi = jnp.arccos(1-2*(ind+0.5)/(bsize_true*2))
		theta = jnp.pi*(ind+0.5)*(gr)*2
		def norm_fun_1(theta):
			theta = theta - 2*jnp.pi
			return theta
		def norm_cond_1(theta):
			return theta > jnp.pi*2
		
		def norm_fun_2(theta):
			theta = theta + 2*jnp.pi
			return theta
		def norm_cond_2(theta):
			return theta < 0
			
		theta = jax.lax.while_loop(norm_cond_1,norm_fun_1,theta)
		theta = jax.lax.while_loop(norm_cond_2,norm_fun_2,theta)
		sgrid = sgrid.at[ind].set(orad*jnp.array([jnp.cos(theta)*jnp.sin(phi),jnp.sin(theta)*jnp.sin(phi),jnp.cos(phi)-1]))
		return sgrid, ind
		
	sgrid,_ = jax.lax.scan(ball_fun_3,sgrid,np.arange(bsize))
	return sgrid
	
#A function that returns an array of points on a disk using a fibbonacci spiral lattice
@partial(jax.jit,static_argnums=(0,1))
def create_disk_grid(bsize,rad):
	sgrid = jnp.zeros((bsize,2))
	gr = (1+jnp.sqrt(5))/2
	def ball_fun_3(sgrid,ind):
		phi = rad*jnp.sqrt((ind+1)/(bsize))
		theta = jnp.pi*(ind+0.5)*(gr)*2
		def norm_fun_1(theta):
			theta = theta - 2*jnp.pi
			return theta
		def norm_cond_1(theta):
			return theta > jnp.pi*2
		
		def norm_fun_2(theta):
			theta = theta + 2*jnp.pi
			return theta
		def norm_cond_2(theta):
			return theta < 0
			
		theta = jax.lax.while_loop(norm_cond_1,norm_fun_1,theta)
		theta = jax.lax.while_loop(norm_cond_2,norm_fun_2,theta)
		sgrid = sgrid.at[ind].set(jnp.array([phi,theta]))
		return sgrid, ind
		
	sgrid,_ = jax.lax.scan(ball_fun_3,sgrid,np.arange(bsize))
	return sgrid
	
#A function that returns an array of points on a disk using a fibbonacci spiral lattice
@partial(jax.jit,static_argnums=(0,1))
def create_disk_grid_charg(bsize,rad):
	sgrid = jnp.zeros((bsize,3))
	gr = (1+jnp.sqrt(5))/2
	def ball_fun_3(sgrid,ind):
		phi = rad*jnp.sqrt((ind+1)/(bsize))
		theta = jnp.pi*(ind+0.5)*(gr)*2
		def norm_fun_1(theta):
			theta = theta - 2*jnp.pi
			return theta
		def norm_cond_1(theta):
			return theta > jnp.pi*2
		
		def norm_fun_2(theta):
			theta = theta + 2*jnp.pi
			return theta
		def norm_cond_2(theta):
			return theta < 0
			
		theta = jax.lax.while_loop(norm_cond_1,norm_fun_1,theta)
		theta = jax.lax.while_loop(norm_cond_2,norm_fun_2,theta)
		sgrid = sgrid.at[ind].set(jnp.array([phi*jnp.cos(theta),phi*jnp.sin(theta),0]))
		return sgrid, ind
		
	sgrid,_ = jax.lax.scan(ball_fun_3,sgrid,np.arange(bsize))
	return sgrid

#A fucntion that reads a martini file to get interaction strengths
def get_int_strength(bead_1,bead_2,martini_file):
	if(bead_2 == "GHOST"):
		return 0
	string = " "*(6-len(bead_1))+bead_1+" "*(6-len(bead_2))+bead_2
	string2 = " "*(6-len(bead_2))+bead_2+" "*(6-len(bead_1))+bead_1
	mfile = open(martini_file,"r")
	content = mfile.readlines()
	for i,line in enumerate(content):
		if(string in line or string2 in line):
			return -float(line[32:45])
			
def get_mem_def(martini_file):
	#We use interactions strengths from martini using a POPE(Q4p)/POPG(P4)/POPC(Q1)? lipid as a template
	W_B_mins = jnp.array([get_int_strength("W",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	LH1_B_mins = jnp.array([get_int_strength("P4",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	LH2_B_mins = jnp.array([get_int_strength("Q5",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	LH3_B_mins = jnp.array([get_int_strength("SN4a",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	LH4_B_mins = jnp.array([get_int_strength("N4a",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	LT1_B_mins = jnp.array([get_int_strength("C1",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	LT2_B_mins = jnp.array([get_int_strength("C4h",list(Beadtype.keys())[i],martini_file) for i in range(len(Beadtype.keys()))])
	Charge_B_mins =jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype="float64")
															   # #
	return (W_B_mins,LH1_B_mins,LH2_B_mins,LH3_B_mins,LH4_B_mins,LT1_B_mins,LT2_B_mins,Charge_B_mins)	

#Ploting some graphs of local minima. This is more complex than a simple plot as
#The information is on a spiral lattice.
def create_graphs(orient_dir,col_grid,pot_grid,angs,resa):
	graph_mesh_cols = jnp.zeros((resa,resa,4))
	graph_mesh_vals = jnp.zeros((resa,resa,2))
	
	
	def get_ind(resa,angs):
		ang1 = angs[0]
		ang2 = angs[1]
		ind1 = resa*ang1/(jnp.pi/2)
		ind2 = resa*ang2/(2*jnp.pi)
		ind1 = jnp.floor(ind1)
		ind2 = jnp.floor(ind2)
		ind1 = jnp.array(ind1,dtype=int)
		ind2 = jnp.array(ind2,dtype=int)
		return ind1,ind2
	def set_vals(graph_mesh,vals,leng):
		graph_mesh = graph_mesh.at[:,:,-1].set(1e-5)
		def set_vals_fun_1(graph_mesh,ind):
			ind1,ind2 = get_ind(resa,angs[ind])
			graph_mesh = graph_mesh.at[ind1,ind2,:-1].set(graph_mesh[ind1,ind2,:-1]+vals[ind])
			graph_mesh = graph_mesh.at[ind1,ind2,-1].set(graph_mesh[ind1,ind2,-1]+1)
			return graph_mesh,ind
		graph_mesh,_ = jax.lax.scan(set_vals_fun_1,graph_mesh,jnp.arange(angs.shape[0]))
		def div_fun_1(d_graph_mesh,ind):
			def div_fun_2(ind_fix,ind):
				divided = graph_mesh[ind_fix,ind,:-1]/(graph_mesh[ind_fix,ind,-1]+1e-9)
				def empty(divided):
					return jnp.zeros(leng)#TODO add  nan
				def nempty(divided):
					return divided
				divided = jax.lax.cond(graph_mesh[ind_fix,ind,-1] > 0.5, nempty,empty,divided)
				return ind_fix,divided
			_,row = jax.lax.scan(div_fun_2,ind,jnp.arange(resa))
			d_graph_mesh = d_graph_mesh.at[ind].set(row)
			return d_graph_mesh,ind
		
		final_mesh,_ = jax.lax.scan(div_fun_1,jnp.zeros((resa,resa,leng)),jnp.arange(resa))
		return final_mesh
	graph_mesh_cols = set_vals(graph_mesh_cols,col_grid,3)
	graph_mesh_vals = set_vals(graph_mesh_vals,pot_grid,1)
	plt.imshow(np.array(graph_mesh_cols),extent=[0,jnp.pi*2,0,jnp.pi/2],aspect=4)
	plt.xlabel("Theta")
	plt.ylabel("Phi")
	plt.title("Orientation of minima given starting position (z,theta,phi)")
	plt.tight_layout()
	plt.savefig(orient_dir+"local_minima_orientation.png")
	plt.imshow(graph_mesh_vals,extent=[0,jnp.pi*2,0,jnp.pi/2],aspect=4)
	plt.xlabel("Theta")
	plt.ylabel("Phi")
	plt.title("Relative potential energy of minima given starting position (z,theta,phi)")
	plt.tight_layout()
	plt.savefig(orient_dir+"local_minima_potential.png")
	plt.clf()
	
#concerts three letter codes to one for transmembrane residue output
def three2one(s):
	if(s == "ALA"):
		return "A"
	elif(s == "ARG"):
		return "R"
	elif(s=="LYS"):
		return "K"
	elif(s == "ASN"):
		return "N"
	elif(s == "ASP"):
		return "D"
	elif(s == "CYS"):
		return "C"
	elif(s == "GLU"):
		return "E"
	elif(s == "GLN"):
		return "Q"
	elif(s == "GLY"):
		return "G"
	elif(s == "HIS"):
		return "H"
	elif(s == "ILE"):
		return "I"
	elif(s == "LEU"):
		return "L"
	elif(s == "MET"):
		return "M"
	elif(s == "PHE"):
		return "F"
	elif(s == "PRO"):
		return "P"
	elif(s == "SER"):
		return "S"
	elif(s == "THR"):
		return "T"
	elif(s == "TRP"):
		return "W"
	elif(s == "TYR"):
		return "Y"
	elif(s == "VAL"):
		return "V"
	else:
		return "X"
	
#formatting function	
def form(val):
	new_val = np.format_float_positional(val,precision=3)
	return new_val

def shifted_cos_cutoff(x,deg):
	ret_val = 0.5*(1.0+jnp.cos(jnp.pi*x/deg))
	def lzero(ret_val):
		return 1.0
	def gzero(ret_val):
		return ret_val
		
	ret_val = jax.lax.cond(x<0,lzero,gzero,ret_val)
	
	def gone(ret_val):
		return 0.0
	def lone(ret_val):
		return ret_val
		
	ret_val = jax.lax.cond(x>deg,gone,lone,ret_val)
	return ret_val


def pts_around(pointsA,points,rng):
	hitpts = jnp.zeros(points.shape[0],dtype=int)
	def a1fun(hitpts,ind):
		ind_fix=ind
		def a2fun(hitpts,ind):
			def in_range(hitpts):
				hitpts = hitpts.at[ind_fix].set(1)
				return hitpts
			def nin_range(hitpts):
				return hitpts
			hitpts = jax.lax.cond(jnp.linalg.norm(points[ind_fix]-pointsA[ind])<rng,in_range,nin_range,hitpts)
			return hitpts,ind
		hitpts,_=jax.lax.scan(a2fun,hitpts,jnp.arange(pointsA.shape[0]))
		return hitpts,ind
	hitpts,_=jax.lax.scan(a1fun,hitpts,jnp.arange(points.shape[0]))
	return hitpts

def pad_num(arrays):
	pos_num = arrays.shape[0]
	no_per_cpu = (pos_num-1)//no_cpu
	no_per_cpu += 1
	new_num = no_per_cpu*no_cpu
	return new_num
	
	
def SphereGridN(den,end_ang,orad):
	grid_points = []
	direcs = []
	frden = int(orad*den*2*(end_ang)/np.pi)+1
	rings = np.linspace(0,end_ang,frden)
	for xr,r in enumerate(rings):
		rad = np.sin(r)*orad
		rden = int(rad*den)
		new_ring = np.linspace(0,np.pi*2,4*rden+2)[:-1]
		rrand = np.random.rand()
		for nr in new_ring:
			ang1 = xr*rrand+nr
			grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,orad-orad*np.cos(r)])
			direcs.append([np.cos(ang1)*np.sin(r),np.sin(ang1)*np.sin(r),np.cos(r)])

	return np.array(grid_points),np.array(direcs)

def DiskGrid(den,orad):
	grid_points = []
	direcs = []
	frden = int(den*2*(orad)/np.pi)
	rings = np.linspace(0,orad,frden)
	for xr,r in enumerate(rings[1:]):
		rad = r
		rden = int(rad*den)+1
		new_ring = np.linspace(0,np.pi*2,4*rden)[:-1]
		rrand = np.random.rand()
		for nr in new_ring:
			ang1 = xr*rrand+nr
			grid_points.append([np.cos(ang1)*rad,np.sin(ang1)*rad,0])
			direcs.append([0,0,1])
	return np.array(grid_points),np.array(direcs)
	
def check_for_nan(val,pnum):
	def isnanj():
		return 1
		jax.debug.print("Nan Check {x}",x=pnum)
	def isnnanj():
		return 0
	return jax.lax.cond(jnp.isnan(val),isnanj,isnnanj)
	
	
	
def get_box_slice(points,p,r):
	indices = np.arange(points.shape[0])
	islice = indices[points[:,0]>p[0]-r[0]]
	pslice = points[points[:,0]>p[0]-r[0]]
	islice = islice[pslice[:,0]<p[0]+r[0]]
	pslice = pslice[pslice[:,0]<p[0]+r[0]]
	islice = islice[pslice[:,1]>p[1]-r[1]]
	pslice = pslice[pslice[:,1]>p[1]-r[1]]
	islice = islice[pslice[:,1]<p[1]+r[1]]
	pslice = pslice[pslice[:,1]<p[1]+r[1]]
	islice = islice[pslice[:,2]>p[2]-r[2]]
	pslice = pslice[pslice[:,2]>p[2]-r[2]]
	islice = islice[pslice[:,2]<p[2]+r[2]]
	pslice = pslice[pslice[:,2]<p[2]+r[2]]
	return pslice,islice
	
def get_binned_points(points,rad):
	binned_points = np.zeros((points.shape[0],points.shape[0]),dtype=int)
	max_in_rad = 0	
	for i,p in enumerate(points):
		_,box = get_box_slice(points,p,[rad,rad,rad])
		binned_points[i][:box.shape[0]] = box
		binned_points[i][box.shape[0]:]=box[-1]
		if(box.shape[0] > max_in_rad):
			max_in_rad = box.shape[0]
	return binned_points[:,:max_in_rad],max_in_rad   


@jax.jit
def get_sph_circ(grid,cen,angle):
	n = 20
	dx = jnp.pi/n
	dy = jnp.pi/n
	xs = jnp.linspace(0,2*jnp.pi-dx,2*n)+dx/2
	ys = jnp.linspace(0,jnp.pi-dy,n)+dy/2
	def scloop1(grid,ind):
		ind1 = ind
		def scloop2(grid,ind):
			dist = jnp.arccos(jnp.cos(cen[0])*jnp.cos(xs[ind1])+jnp.sin(cen[0])*jnp.sin(xs[ind1])*jnp.cos(cen[1]-ys[ind]))
			def langle(grid):
				grid = grid.at[ind1,ind].set(dx*(jnp.cos(ys[ind]-dy/2)-jnp.cos(ys[ind]+dy/2)))
				return grid
			def gangle(grid):
				return grid
			grid = jax.lax.cond(dist<angle,langle,gangle,grid)
			return grid,ind
		grid,_ = jax.lax.scan(scloop2,grid,jnp.arange(n))
		return grid,ind
	grid,_ = jax.lax.scan(scloop1,grid,jnp.arange(2*n))
	return grid
	
def add_Reses(Res, Resitp):
    current_res_no = len(Reses)
    current_beadtype_no = len(Beadtype)
    current_bead_no = len(Beads)
    
    Reses[Res] = current_res_no
    print("Initial Beads:", Beads)
 
    with open(Resitp, "r") as itpfile:
        itp_lines = itpfile.readlines()
 
    restobead_add = []
    for iline in itp_lines:
        iline = iline.strip()
        if iline.startswith(";"):
            continue  # Skip comment lines
        if Res in iline:
            sline = iline.split()
            if len(sline) >= 7:
                bt = sline[1]
                bead = sline[4]
                
                restobead_add.append(bt)
                
                # Add bead if it's new
                if bead not in Beads:
                    Beads[bead] = current_bead_no
                    current_bead_no += 1
                else:
                    #continue
                    print(f"Bead '{bead}' already exists with index {Beads[bead]}")
                
                # Add bead type if it's new
                if bt not in Beadtype:
                    Beadtype[bt] = current_beadtype_no
                    current_beadtype_no += 1

    ResToBead.append(restobead_add)

    
def add_AtomToBeads(fn):
	bead_contents = []
	bead_names = []
	beads = []
	first_bead = True
	lfile = open(os.path.join(fn),"r")
	content = lfile.read()
	lfile.close()
	content = content.split("\n")
	for c in content:
		if("[" not in c and c[:4]=="ATOM"):	
			bead = c[12:17].strip()
			beads.append(bead)
		else:
			bead_names.append(c[1:-1])
			if first_bead:
				first_bead = False
			else:
				bead_contents.append(beads)
				beads = []
	bead_contents.append(beads)
	AtomsToBead.append(bead_contents)
	return bead_names,bead_contents


