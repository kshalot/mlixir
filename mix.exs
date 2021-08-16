defmodule Mlixir.MixProject do
  use Mix.Project

  def project do
    [
      app: :mlixir,
      version: "0.1.0",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
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
      {:benchee, "~> 1.0", only: :dev},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "nx"},
      {:scidata, "~> 0.1.1"},
    ]
  end
end
