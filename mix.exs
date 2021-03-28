defmodule Mlixir.MixProject do
  use Mix.Project

  def project do
    [
      app: :mlixir,
      version: "0.1.0",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:credo, "~> 1.5", only: [:dev, :test], runtime: false},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "feat/add-matrix-inversion", sparse: "nx"}
    ]
  end
end
