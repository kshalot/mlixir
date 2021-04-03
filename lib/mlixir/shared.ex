defmodule Mlixir.Shared do
  @moduledoc """
  Mlixir shared utilities.
  """

  import Nx.Defn

  @doc """
  Left pads a tensor with the given scalar.
  """
  defn left_pad(t, value) do
    config = transform(Nx.rank(t), & List.duplicate({0, 0, 0}, &1 - 1) ++ [{1, 0, 0}])
    Nx.pad(t, value, config)
  end
end
