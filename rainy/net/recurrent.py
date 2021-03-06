from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Generic, Iterable, Optional, Sequence, Tuple, Union, TypeVar
from .init import lstm_bias, Initializer
from ..prelude import Self
from ..utils import Device

from block_wrapper import BlockWrapper

class RnnState(ABC):
    @abstractmethod
    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        pass

    @abstractmethod
    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        pass

    @abstractmethod
    def fill_(self, f: float) -> None:
        pass

    @abstractmethod
    def mul_(self, x: Tensor) -> None:
        pass


RS = TypeVar('RS', bound=RnnState)


class RnnBlock(Generic[RS], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, hidden: RS, masks: Optional[Tensor]) -> Tuple[Tensor, RS]:
        pass

    @abstractmethod
    def initial_state(self, batch_size: int, device: Device) -> RS:
        pass


def _apply_mask(mask: Optional[Tensor], *args) -> Sequence[Tensor]:
    if mask is None:
        return tuple(map(lambda x: x.unsqueeze(0), args))
    else:
        m = mask.view(1, -1, 1)
        return tuple(map(lambda x: x * m, args))


def _reshape_batch(x: Tensor, mask: Optional[Tensor], nsteps: int) -> Tuple[Tensor, Tensor]:
    x = x.view(nsteps, -1, x.size(-1))
    if mask is None:
        return x, torch.ones_like(x[:, :, 0])
    else:
        return x, mask.view(nsteps, -1)


def _haszero_iter(mask: Tensor, nstep: int) -> Iterable[Tuple[int, int]]:
    has_zeros = (mask[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
    if has_zeros.dim() == 0:
        haszero = [has_zeros.item() + 1]
    else:
        haszero = (has_zeros + 1).tolist()
    return zip([0] + haszero, haszero + [nstep])


class LstmState(RnnState):
    def __init__(self, h: Tensor, c: Tensor, squeeze: bool = True) -> None:
        self.h = h
        self.c = c
        if squeeze:
            self.h.squeeze_(0)
            self.c.squeeze_(0)

    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        return LstmState(self.h[x], self.c[x])

    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        self.h[x] = value.h[x]
        self.c[x] = value.c[x]

    def fill_(self, f: float) -> None:
        self.h.fill_(f)
        self.c.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)
        self.c.mul_(x)


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(bias_init = lstm_bias()),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, **kwargs)
        initializer(self.lstm)
        print('using lstm block!')

    def forward(
            self,
            x: Tensor,
            hidden: LstmState,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, LstmState]:
        in_shape = x.shape
        if in_shape == hidden.h.shape:
            out, (h, c) = self.lstm(x.unsqueeze(0), _apply_mask(mask, hidden.h, hidden.c))
            return out.squeeze(0), LstmState(h, c)
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask, nsteps)
        res, h, c = [], hidden.h, hidden.c
        for start, end in _haszero_iter(mask, nsteps):
            m = mask[start].view(1, -1, 1)
            processed, (h, c) = self.lstm(x[start:end], (h * m, c * m))
            print('h c min max', h.min(), h.max(), c.min(), c.max())
            res.append(processed)
        return torch.cat(res).view(in_shape), LstmState(h, c)

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        zeros = device.zeros((batch_size, self.input_dim))
        return LstmState(zeros, zeros, squeeze=False)


class GruState(RnnState):
    def __init__(self, h: Tensor) -> None:
        self.h = h

    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        return GruState(self.h[x])

    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        self.h[x] = value.h[x]

    def fill_(self, f: float) -> None:
        self.h.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)


class GruBlock(RnnBlock[GruState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        #self.gru = nn.GRU(input_dim, output_dim, **kwargs)
        #initializer(self.gru)

        self.gru = BlockWrapper(input_dim, output_dim, output_dim, **kwargs)
        initializer(self.gru.myrnn.block_lstm)

    def forward(
            self,
            x: Tensor,
            hidden: GruState,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, GruState]:
        in_shape = x.shape
        #print('compare in/h', in_shape, hidden.h.shape)
        if in_shape[0] == hidden.h.shape[0]:
            #print('inp shape', x.unsqueeze(0).shape, 'h shape', (_apply_mask(mask, hidden.h))[0].shape)
            out, h = self.gru(x.unsqueeze(0), *_apply_mask(mask, hidden.h))
            #print('out shape', out.shape, 'h shape', h.shape)
            return out.squeeze(0), GruState(h.squeeze_(0))
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask, nsteps)
        res, h = [], hidden.h
        for start, end in _haszero_iter(mask, nsteps):
            #print('inp shape', x[start:end].shape, 'h  loop shape', h.shape)
            processed, h = self.gru(x[start:end], h * mask[start].view(1, -1, 1))
            #print('h min max', h.min(), h.max())
            #print('process shape', processed.shape, 'h shape after', h.shape)
            res.append(processed)
        return torch.cat(res).view(in_shape), GruState(h.squeeze_(0))

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((batch_size, self.input_dim)))

class DummyState(RnnState):
    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        return self

    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        pass

    def fill_(self, f: float) -> None:
        pass

    def mul_(self, x: Tensor) -> None:
        pass


class DummyRnn(RnnBlock[DummyState]):
    DUMMY_STATE = DummyState()

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)

    def forward(
            self,
            x: Tensor,
            hidden: DummyState = DUMMY_STATE,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, DummyState]:
        return x, hidden

    def initial_state(self, batch_size: int, device: Device) -> DummyState:
        return self.DUMMY_STATE
