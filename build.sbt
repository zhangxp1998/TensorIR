name := "TensorIR"

scalaVersion := "2.12.8"

resolvers += Resolver.sonatypeRepo("snapshots")

// cps

autoCompilerPlugins := true

addCompilerPlugin("org.scala-lang.plugins" % "scala-continuations-plugin_2.12.0" % "1.0.3")

libraryDependencies += "org.scala-lang.plugins" % "scala-continuations-library_2.12" % "1.0.3"

scalacOptions += "-P:continuations:enable"

// libraryDependencies += "org.scala-lang.lms" %% "lms-core-macrovirt" % "0.9.0-SNAPSHOT"

libraryDependencies += "org.scalatest" % "scalatest_2.12" % "3.0.4"

libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value % "compile"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value % "compile"

libraryDependencies ++= Seq(
    "com.github.vagmcs" %% "optimus" % "3.2.0",
    "com.github.vagmcs" %% "optimus-solver-oj" % "3.2.0",
    "com.github.vagmcs" %% "optimus-solver-lp" % "3.2.0"
)

autoCompilerPlugins := true

val paradiseVersion = "2.1.0"

addCompilerPlugin("org.scalamacros" % "paradise" % paradiseVersion cross CrossVersion.full)

// tests are not thread safe
parallelExecution in Test := false



sourceDirectory in Compile := (baseDirectory( _ / "src" )).value

unmanagedSourceDirectories in Compile += (baseDirectory( _ / "test" )).value


lazy val lms = ProjectRef(file("./lms-clean"), "lms-clean")
lazy val tutorials = (project in file(".")).dependsOn(lms)
