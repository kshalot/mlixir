defmodule Mlixir.Model do
  # FIXME Revise this API - it's impossible to use things like
  # options and still fully adhere to it. Probably best to replace
  # with sth like a protocol + struct (maybe?)

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
