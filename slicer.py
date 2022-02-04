from typing import Tuple,Union,List
from collections.abc import Iterable
from itertools import zip_longest
from copy import deepcopy


class MultiSliceList:

    def __init__(self, alist):
        self.seq = alist
        self.__deep = self.__depth(self.seq)
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
            none_slice = slice(None,None,None)
            last_valid = None
            for k in key[:-1]:
                if k != none_slice:
                    last_valid = k
                ref = ref.__getitem__(eval(f"{k}"))

            if isinstance(key[-1],slice):

                if key[-1] == none_slice:
                    ref[:] = value
                    self.seq[last_valid][:] = ref

                else:
                    ref[key[-1]] = value

                self.seq[last_valid][:] = ref
                    
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

    def __getitem__(self, key):

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
            return self.seq[key]

        raise TypeError(f"{type(self.seq).__name__} indices must be integers or slices")


    def __add__(self, other):

        if not isinstance(other, (list,MultiSliceList)):
            raise ValueError("Invalid operand type must be list or MultiSliceList instance")

        if isinstance(other, list):
            return MultiSliceList(self.seq + other)

        if isinstance(other, MultiSliceList):
            return MultiSliceList(self.seq+other.seq)

    def __mul__(self, scalar):

        if not isinstance(scalar, (float, int)):
            raise ValueError("Invalid operand type must be int or float")

        return MultiSliceList([e*scalar for e in self.seq])

    def __truediv__(self, scalar):

        if not isinstance(scalar, (float, int)):
            raise ValueError("Invalid operand type must be int or float")

        if scalar == 0:
            raise ZeroDivisionError

        return MultiSliceList([e/scalar for e in self.seq])

    def __floordiv__(self, scalar):

        if not isinstance(scalar, (float,int)):
            raise ValueError("Invalid operand type must be int or float")

        if scalar == 0:
            raise ZeroDivisionError

        return MultiSliceList([e//scalar for e in self.seq])

    def __mod__(self, base):

        if not isinstance(base, (float,int)):
            raise ValueError("Invalid operand type must be int or float")

        if base == 0:
            raise ZeroDivisionError

        return MultiSliceList([e%base for e in self.seq])

    def __pow__(self, power):

        if not isinstance(power,(float,int)):
            raise ValueError("Invalid operand type must be int or float")

        return MultiSliceList([e**power for e in self.seq])
    

    def __depth(self, seq: List):
        if isinstance(seq, list):
            return 1 + max(self.__depth(li) for li in seq)
        return 0

    def __flatten(self, seq: List):
        if type(seq) != list:
            return [seq]
        else:
            return sum([self.__flatten(x) for x in seq],[])
    
    @property
    def deep(self) -> int:
        return self.__deep

    @property
    def T(self) -> int:
        if self.deep == 1:
            return MultiSliceList([[n] for n in self])
        else:
            return MultiSliceList([list(r) for r in zip(*self.seq)])

    @property
    def flatten(self):
        return MultiSliceList(self.__flatten(self.seq))

    def extend(self, other, inplace: bool=False):
        """extend MultiSliceList instance, similar to list's extend.

           params: other - list or MultiSliceList instance to extend self by
                   fit - optionally extend self by other with adherence to current depth
                   inplace - optionally extend self by other inplace

           return either MultiSliceList or None depending on inplace
        """

        if not isinstance(other,(list,MultiSliceList)):
            raise ValueError("Invalid operand type must be list or MultiSliceList")

        if self.deep:

            if not inplace:
                return self + [other]
            else:
                self[:] = self + [other]
        else:
            if not inplace:
                return self + other
            else:
                self[:] = self + other

    def filter(self, function, when: bool=True, inplace: bool=False):
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
            args = iter(self.flatten)
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


    def zip(self, groupby: int=None, inplace: bool=False):
        """
        returns a zipped representation of the list with the ith pairing of each sublist
        inside of self. optionally if inplace, then changes to self are made in place.
        if k is set, zip will maximize k-length pairings.

        params: groupby - int, the size of each grouping desired
                inplace - bool, if True will update self inplace
        """

        def tight(iterable):
            nonlocal groupby
            args = [iter(iterable)] * groupby
            for g in zip_longest(*args,fillvalue=False):
                while not all(g := list(g)): g.pop()
                yield list(g)

        res = None
        if groupby is None:
            res = MultiSliceList([list(p) for p in zip(*self)])
        else:
            res = MultiSliceList([p for p in tight(self.__flatten(self.seq))])

        if inplace:
            self[:] = res
        else:
            return res


    def applymap(self, function, flattened: bool=False,inplace: bool=False):
        """
        maps the given function onto self, if inplace checked, updates inplace

        params: function - function, to call on each element inside self
                flattened - bool, flattens self, then applies function 
                inplace - bool, if True will update self inplace
        """

        def mapper():
            nonlocal function, flattened
            args = iter(self.flatten) if flattened else iter(self)
            while True:
                try:
                    yield function(next(args))
                except StopIteration:
                    break

        try:
            if inplace:
                self[:] = MultiSliceList(list(mapper()))
            else:
                return MultiSliceList(list(mapper()))
        except TypeError as e:
            print(e)
                


        


    
