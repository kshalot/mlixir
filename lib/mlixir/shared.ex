defmodule Mlixir.Shared do
  @moduledoc """
  Mlixir shared utilities.
  """

  import Nx.Defn

  @doc """
  Left pads a tensor with the given scalar.
  """
  defn left_pad(t, value \\ 0) do
    Nx.pad(t, value, [{0, 0, 0}, {1, 0, 0}])
  end
end
