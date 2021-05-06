defmodule Mlixir.KNN do
  @moduledoc """
  Model representing a k-nearest neighbours classifier.
  """

  import Nx.Defn

  @behaviour Mlixir.Model

  @doc """
  """
  @impl true
  defn fit(x, y) do
    Nx.stack(x, y)
  end

  @doc """
  """
  @impl true
  defn predict(model, x) do
  end
end
