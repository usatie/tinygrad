# [Bounty] Remove realize from __setitem__ and get TestSetitemLoop.test_arange to be one kernel

## Overview

This PR optimizes setitem operations by removing the `realize()` call from the `__setitem__` method in the Tensor class. 

- [x] integer indexing
- [x] slice indexing
- [x] ellipsis indexing
- [x] None indexing
- [ ] Tensor indexing (not supported yet, but properly handled by existing code `_getitem`)

Pseudo code for the new `__setitem__` implementation is as follows:

```python
def __setitem__(self, indices, v):
  mask = self._generate_setitem_mask(indices)
  padded_v - self._generate_setitem_padded_value(indices, v)
  res = mask.where(padded_v, self)
  self.assign(res)
```

## Performance Improvement

**TestSetitemLoop.test_arange benchmark** with `RANGEIFY=1`:
- **Master branch**: 20 kernels scheduled in total
- **This PR**: 3 kernels  
- **Command**: `DEBUG=1 RANGEIFY=1 python -m pytest test/test_setitem.py::TestSetitemLoop -rP`
- **Improvement**: 85% reduction in kernel count

This optimization significantly benefits scenarios with repeated setitem operations, such as tensor initialization loops and in-place tensor modifications.

## ❌ Failing Test Case

**Test**: `test_setitem.py::TestSetitem::test_setitem_inplace_mul`  
**Error**: `AssertionError: must be BUFFER Ops.BUFFER_VIEW`

**Root cause**: The `assign` method cannot properly handle tensor views (even contiguous slices) during buffer realization. When inplace operations like `*=` call `assign` on slice views, the underlying buffer system fails because it encounters a `BUFFER_VIEW` operation instead of the expected `BUFFER` operation.

**Technical details**:
```
t[:3] = t[:3] * 10 # This works fine
t[:3] *= 10        # But this causes a crash when realized
```
- Python interpreter internally translates the expression like this `t[:3] *= 10` into the code below:
```python
_slice = t[:3]
_slice += 10
t[:3] = _slice
```
- During realization, the buffer property assertion fails: `assert self.op is Ops.BUFFER`  
- The system encounters `Ops.BUFFER_VIEW` instead of `Ops.BUFFER`, causing the crash

**Conflict**: This test can be fixed by changing `__imul__` to use `replace` instead of `assign`, but then `test_assign.py::TestAssign::test_permuted_assignment` would fail because it expects inplace operations to properly reject non-contiguous tensors.

## Possible Solutions

I'm considering these approaches, but unsure which is best. Are there better alternatives?

1. **Fix assign for views**: Modify `assign` method to detect slice views and handle them appropriately  
2. **Fix buffer realization**: Update the `realize()` pipeline to properly handle `BUFFER_VIEW` operations

