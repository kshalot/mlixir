defmodule Mlixir.Model do
  @moduledoc """
  Behaviour defining the public API of every Mlixir model.
  """

  @doc """
  Train the model.
  """
  @callback fit(Nx.Tensor.t, Nx.Tensor.t) :: Nx.Tensor.t

  @doc """
  Predict using the model.
  """
  @callback predict(Nx.Tensor.t, Nx.Tensor.t) :: Nx.Tensor.t
end
