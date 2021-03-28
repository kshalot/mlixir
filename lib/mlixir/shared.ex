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

  @doc """
  Creates a row vector of shape {n, 1} from a given column vector of shape {n}.
  """
  defn row_vector(v) do
    {v_length} = Nx.shape(v)
    Nx.reshape(v, {v_length, 1})
  end
end
