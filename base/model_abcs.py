from abc import ABC, abstractmethod


class EMModel(ABC):
    @abstractmethod
    def _Estep(self, **data_kwargs):
        """Estimates posterior distribution P(X|Z, theta)."""

    @abstractmethod
    def _Mstep(self, **data_kwargs):
        """Updates Parameters."""

    @abstractmethod
    def fit(self, **data_kwargs):
        """Fit model using EM algo to data."""

    @abstractmethod
    def predict(self, **data_kwargs):
        """Make predictions for data_kwargs."""


class MixtureModel(EMModel):
    @abstractmethod
    def __len__(self):
        """Returns number of mixtures."""

    @abstractmethod
    def __getitem__(self, idx):
        """Returns tuple (priors[idx], params[idx])."""
