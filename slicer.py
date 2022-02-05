from typing import Tuple,Union,List,TypeVar
from collections.abc import Iterable
import itertools
import copy
import inspect
import random

MSL = TypeVar('MultiSliceList',bound=object)

class MultiSliceList(object):

    def __init__(self, alist=None):
        self.seq = alist if isinstance(alist,list) else (list(alist) if alist else list())
        self.__deep = self.__depth(self.seq) if self.seq else 0 
        self.__index = -1

    def __repr__(self):
        return f"MultiSliceList([{','.join(str(v) for v in self.seq)}])"

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.seq)

    def __next__(self):
        self.__index += 1
        if self.__index >= len(self.seq):
            self.__index = -1
            raise StopIteration
        else:
            return self.seq[self.__index]

    def __setitem__(self, key, value):

        # key is multi-index tuple
        if isinstance(key,tuple):

            if len(key) > self.__deep:
                raise IndexError(f"Invalid indices for MultiSliceList of depth {self.__deep}")

            ref = self.seq.copy()
            last_valid = None
            for k in key[:-1]:
                last_valid = k
                ref = ref.__getitem__(k)

            if isinstance(key[-1],slice):

                if key[-1] == slice(None,None,None):
                    self.seq[last_valid][:] = value

                else:
                    self.seq[last_valid][key[-1]] = value
                    
            else:
                
                if hasattr(ref[key[-1]],'__iter__'):
                    ref[key[-1]][:] = value
                else:
                    ref[key[-1]] = value

                self.seq[:key[-1]][:] = ref
                
        # key is single index ~ slice or int
        if isinstance(key,(slice,int)):
            self.seq[key] = value

        # case of assignment to other MultiSliceList (from operands)
        if isinstance(value,MultiSliceList):
            self = value
            self.__deep = value.deep
        else:
            # update dimension
            self.__deep = self.__depth(self.seq)

    def __getitem__(self, key) -> MSL:

        # key is multi-index
        if isinstance(key,tuple):

            # ex: indexing 3 dimensions when there's only depth of 2
            if len(key) > self.__deep:
                raise IndexError(f"Invalid indices for MultiSliceList of depth {self.__deep}")

            # get value & return 
            val = self.seq.copy()
            for k in key:
                val = val[k]
            return val

        # key is single index ~ slice or int
        if isinstance(key,(slice,int)):
            return MultiSliceList(self.seq[key])

        raise TypeError(f"{type(self.seq).__name__} indices must be integers or slices")


    def __add__(self, other) -> MSL:

        if not isinstance(other, (list,MultiSliceList)):
            raise ValueError("Invalid operand type must be list or MultiSliceList instance")

        if isinstance(other, list):
            return MultiSliceList(self.seq + other)

        if isinstance(other, MultiSliceList):
            return MultiSliceList(self.seq+other.seq)

    def __mul__(self, scalar) -> MSL:

        if not isinstance(scalar, (float, int)):
            raise ValueError("Invalid operand type must be int or float")

        return MultiSliceList([e*scalar for e in self.seq])

    def __truediv__(self, scalar) -> MSL:

        if not isinstance(scalar, (float, int)):
            raise ValueError("Invalid operand type must be int or float")

        if scalar == 0:
            raise ZeroDivisionError

        return MultiSliceList([e/scalar for e in self.seq])

    def __floordiv__(self, scalar) -> MSL:

        if not isinstance(scalar, (float,int)):
            raise ValueError("Invalid operand type must be int or float")

        if scalar == 0:
            raise ZeroDivisionError

        return MultiSliceList([e//scalar for e in self.seq])

    def __mod__(self, base) -> MSL:

        if not isinstance(base, (float,int)):
            raise ValueError("Invalid operand type must be int or float")

        if base == 0:
            raise ZeroDivisionError

        return MultiSliceList([e%base for e in self.seq])

    def __pow__(self, power) -> MSL:

        if not isinstance(power,(float,int)):
            raise ValueError("Invalid operand type must be int or float")

        return MultiSliceList([e**power for e in self.seq])
    

    def __depth(self, seq: list) -> int:
        
        if seq and isinstance(seq, list):
            return 1 + max(self.__depth(li) for li in seq)
        
        return 0

    def __flatten(self, seq: list) -> list:
        
        if type(seq) != list:
            return [seq]
        else:
            return sum([self.__flatten(x) for x in seq],[])
    
    @property
    def deep(self) -> int:
        return self.__deep

    @property
    def T(self) -> MSL:
        if self.deep == 1:
            return MultiSliceList([[n] for n in self])
        else:
            return MultiSliceList([list(r) for r in zip(*self.seq)])

    @property
    def flatten(self) -> MSL:
        return MultiSliceList(self.__flatten(self.seq))

    @property
    def uniq(self) -> MSL:
        return MultiSliceList(set(self.seq))

    def copy(self) -> MSL:
        """
        returns a copy of self
        """
        return copy.deepcopy(self)

    def extend(self, other, inplace: bool=False) -> Union[None,MSL]:
        """extend MultiSliceList instance, similar to list's extend.

           params: other - list or MultiSliceList instance to extend self by
                   inplace - optionally extend self by other inplace

           return either MultiSliceList or None depending on inplace
        """

        if not isinstance(other,(list,MultiSliceList)):
            raise ValueError("Invalid operand type must be list or MultiSliceList")

        if self.deep > 1:

            if not inplace:
                return self + [other]
            else:
                self[:] = self + [other]
        else:
            if not inplace:
                return self + other
            else:
                self[:] = self + other

    def filter(self, function, when: bool=True, flatten: bool=False, inplace: bool=False) -> Union[None,MSL]:
        """
        filter out the values in self by the return value of the function.
        function must return a boolean value. by default, when parameter is True,
        when dictates the delimeter for which the values will be filtered.

        params: function - a boolean returning function
                when - optional filters out the values for wh
                inplace - optional changes the instance in place
        """
        def _filt(cond):
            nonlocal function
            args = iter(self.flatten) if flatten else iter(self)
            while True:
                try:
                    arg = next(args)
                    if (val := function(arg)) == cond:
                        yield arg
                except StopIteration:
                    break

        if inplace:
            self[:] = MultiSliceList([list(_filt(when))])
        else:
            return MultiSliceList([list(_filt(when))])


    def zip(self, groupby: int=None, flatten: bool=False, inplace: bool=False) -> Union[None,MSL]:
        """
        returns a zipped representation of the list with the ith pairing of each sublist
        inside of self. optionally if inplace, then changes to self are made in place.
        if k is set, zip will maximize k-length pairings.

        params: groupby - int, the size of each grouping desired
                inplace - bool, if True will update self inplace
                flatten - bool, if True will flatten self first
        """

        def _tight(iterable):
            nonlocal groupby
            args = [iter(iterable)] * groupby
            for g in itertools.zip_longest(*args,fillvalue=False):
                while not all(g := list(g)): g.pop()
                yield list(g)

        res = None
        if groupby is None:
            res = MultiSliceList([list(p) for p in zip(self)])
        elif flatten:
            res = MultiSliceList([p for p in _tight(self.seq)]).flatten
        else:
            res = MultiSliceList([list(p) for p in _tight(self.seq)])
            
        
        if inplace:
            self.__deep = res.deep
            self[:] = res
        else:
            return res


    def applymap(self, function, inplace: bool=False) -> Union[None,MSL]:
        """
        maps the given function onto self, if inplace checked, updates inplace

        params: function - function, to call on each element inside self
                inplace - bool, if True will update self inplace
        """

        def _mapper():
            nonlocal function
            args = iter(self)
            arity = len(inspect.getfullargspec(function).args)
            while True:
                try:

                    arg = next(args)

                    if hasattr(arg, '__len__'):
                        if len(arg) != arity:
                            raise TypeError("sequence elements must match function arity")
                    elif arity != 1:
                        raise TypeError("sequence elements must match function arity")
                    
                    
                    if hasattr(arg,'__iter__') and arity > 1:
                        yield function(*arg)
                    else:
                        yield function(arg)
                        
                except StopIteration:
                    break
                
        if inplace:
            self[:] = MultiSliceList(list(_mapper()))
        else:
            return MultiSliceList(list(_mapper()))


    def pop(self, index: int=-1) -> None:
        """
        removes element at specified index. if left unspecified, removes last element from self
        """

        if (not self.__deep):
            raise TypeError("unable to pop from empty MultiSliceList")

        self.seq.pop(index)
        if len(self.seq):
            self.__deep = self.__depth(self.seq)
        else:
            self.__deep = 0


    def duplicate(self, k: int=1, inplace: bool=False) -> Union[None,MSL]:
        """
        extends self k times by self. if inplace, all changes are applied to self.
        else returns new MultiSliceList
        """
        if k <= 0 or (not isinstance(k,int)):
            raise ValueError("k must be positive integer")

        if inplace:
            for _ in range(k):
                self.extend(self.seq.copy(),inplace=True)
        else:
            return MultiSliceList(self.seq.copy() for _ in range(k))


    def rotate(self, k: int=1, right: bool=False, inplace: bool=False) -> Union[None,MSL]:
        """
        rotate self k times to the left (by default). optionally, right rotate as well.
        if changes made inplace, return is None; else returns new MultiSliceList
        """

        if inplace:
            
            if right:
                self[:] = self[-k:] + self[:-k]
            else:
                self[:] = self[k:] + self[:k]

        else:

            if right:
                return self[-k:] + self[:-k]
            else:
                return self[k:] + self[:k]


    def sum(self) -> Union[int,float]:
        """
        returns the numerical sum of elements in the MultiSliceList,
        through flattening
        """
        return sum(self.flatten)

    def max(self) -> Union[int,float,list]:
        """
        returns either the numerical maximum of the elements in the MultiSliceList
        or the longest sublist which contains that maximal element
        """

        maxx = max(self.flatten.seq)
        possibles = []
        
        for sub in self:
            if hasattr(sub,'__iter__'):
                if maxx in sub:
                    possibles.append(sub)
            elif maxx == sub:
                possibles.append([sub])
                
        res = sorted(possibles,key=lambda s: len(s))[-1]
        return res if len(res) > 1 else res[0]

    def min(self) -> Union[int,float,list]:
        """
        returns either the numerical minimum of the elements in the MultiSliceList
        or the smallest sublist which contains that minimal element
        """

        minn = min(self.flatten.seq)
        possibles = []

        for sub in self:
            if hasattr(sub,'__iter__'):
                if minn in sub:
                    possibles.append(sub)
            elif minn == sub:
                possibles.append([sub])

        res = sorted(possibles,key=lambda s: len(s))[0]
        return res if len(res) > 1 else res[0]
            

    def clear(self) -> None:
        """
        clears MultiSliceList of all its elements inplace
        """
        self[:] = []

    def combinations(self, r: int=1, replacement: bool=False) -> MSL:
        """
        uses itertools.combinations returns all possible r-sized combinations
        with or without replacement
        """

        if replacement:
            return MultiSliceList(itertools.combinations_with_replacement(self,r))
        else:
            return MultiSliceList(itertools.combinations(self,r))


    def permutations(self, r: int=1, replacement: bool=False) -> MSL:
        """
        uses itertools.permutations returns all possible r-sized permutations
        """

        return MultiSliceList(itertools.permutations(self,r))

    def starmap(self, function, arity_match: bool=True) -> MSL:
        """
        uses itertools.starmap. if arity_match is True, then pairings of
        size which match the arity of the input function will be made (default).
        otherwise, arity of self must match function.
        """

        cp = self.zip(groupby=len(inspect.getfullargspec(function).args))
        if not arity_match:
            cp = self.copy()

        return cp.applymap(function)

    def powerset(self) -> MSL:
        """
        returns the powerset of self
        """
        seq = (itertools.chain
                        .from_iterable(
                            itertools.combinations(self.seq,r) for r in range(len(self)+1)
                            )
               )
        return MultiSliceList(seq)
        
    def choice(self) -> MSL:
        """
        returns a random choice element from self
        """
        return random.choice(self.seq)

    def choices(self, k: int=1, replacement: bool=False) -> MSL:
        """
        returns a sequence of random choice elements from self as MultiSliceList.
        choices can be made optionally with or without replacement.
        """
        if k <= 0:
            raise ValueError("k must be positive integer")

        if not self.deep:
            return self

        if replacement:
            return MultiSliceList(random.choices(self.seq,k=k))
        else:
            return MultiSliceList(random.sample(self.seq,k))

    def counts(self) -> dict:
        """
        returns a dictionary of counts for each element in self
        """

        counter = dict.fromkeys(self,0)
        for k in self:
            counter[k] += 1
        return counter

    def reverse(self, inplace: bool=False) -> Union[None,MSL]:
        """
        optionally reverses self inplace
        """

        if inplace:
            self[:] = self[::-1]
        else:
            return MultiSliceList(self[::-1])
            
    

    
        
    


    
        

    
        

    

        
        
                
        


    
