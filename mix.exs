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
      {:benchee, "~> 1.0"},
      {:nx, "~> 0.2"},
      {:exla, "~> 0.2"},
      {:scidata, "~> 0.1.1"},
    ]
  end
end
