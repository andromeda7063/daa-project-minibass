import Std

/-! Proof kernels for the finite reference problem. No acoustics simulator or
Python program is imported. Analytic envelope assumptions are explicit. -/
namespace MiniBass

def designs {α : Type} : List (List α) → List (List α)
  | [] => [[]]
  | g :: gs => designs gs ++ g.flatMap (fun a => (designs gs).map (fun xs => a :: xs))

def Admissible {α : Type} : List (List α) → List α → Prop
  | [], xs => xs = []
  | g :: gs, xs => Admissible gs xs ∨
      ∃ a, a ∈ g ∧ ∃ ys, Admissible gs ys ∧ xs = a :: ys

theorem enumeration_complete {α : Type} (groups : List (List α)) (xs : List α) :
    xs ∈ designs groups ↔ Admissible groups xs := by
  induction groups generalizing xs with
  | nil => simp [designs, Admissible]
  | cons g gs ih =>
      simp only [designs, List.mem_append, List.mem_flatMap, List.mem_map]
      simp only [ih]
      simp only [Admissible]
      constructor
      · intro h
        rcases h with h | ⟨a, ha, ys, hy, he⟩
        · exact Or.inl h
        · exact Or.inr ⟨a, ha, ys, hy, he.symm⟩
      · intro h
        rcases h with h | ⟨a, ha, ys, hy, he⟩
        · exact Or.inl h
        · exact Or.inr ⟨a, ha, ys, hy, he.symm⟩

theorem feasible_filter {α : Type} (groups : List (List α))
    (feasible : List α → Bool) (xs : List α) :
    xs ∈ (designs groups).filter feasible ↔
      Admissible groups xs ∧ feasible xs = true := by
  simp [List.mem_filter, enumeration_complete]

/- An optimistic predicate can remove candidates only if it is necessary. -/
theorem pruning_sound {α : Type} (candidates : List α) (feasible possible : α → Bool)
    (necessary : ∀ x, feasible x = true → possible x = true) (x : α) :
    x ∈ (candidates.filter possible).filter feasible ↔
      x ∈ candidates.filter feasible := by
  simp only [List.mem_filter]
  constructor
  · rintro ⟨⟨hx, _⟩, hf⟩
    exact ⟨hx, hf⟩
  · rintro ⟨hx, hf⟩
    exact ⟨⟨hx, necessary x hf⟩, hf⟩

/- Conditional analytic prune kernels: n is final nominal amplitude and eta
is final radius. Deriving A/B from complex arithmetic is a separate paper proof. -/
theorem lower_envelope_prune (n A L rho eta : Rat)
    (amplitude_bound : n ≤ A) (radius_bound : rho ≤ eta)
    (lower_needed : L + eta ≤ n) (cut : A < L + rho) : False := by
  grind

theorem upper_envelope_prune (n B U rho eta : Rat)
    (amplitude_bound : B ≤ n) (radius_bound : rho ≤ eta)
    (upper_needed : n + eta ≤ U) (cut : U < B + rho) : False := by
  grind

/- Clauses and quota checks preserve necessary-condition ordering. -/
theorem all_mono {α : Type} (xs : List α) (p q : α → Bool)
    (h : ∀ x, p x = true → q x = true) :
    xs.all p = true → xs.all q = true := by
  simp only [List.all_eq_true]
  intro hp x hx
  exact h x (hp x hx)

theorem quota_mono {α : Type} (xs : List α) (p q : α → Bool)
    (h : ∀ x, p x = true → q x = true) :
    (xs.filter p).length ≤ (xs.filter q).length := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      by_cases hp : p x = true
      · have hq := h x hp
        simp [hp, hq]
        exact ih
      · cases hqx : q x <;> simp [hp, hqx] <;> omega

theorem quota_prune {α : Type} (xs : List α) (p q : α → Bool)
    (h : ∀ x, p x = true → q x = true) (quota : Nat)
    (cut : (xs.filter q).length < quota) :
    ¬ quota ≤ (xs.filter p).length := by
  have hm := quota_mono xs p q h
  omega

/- Exact finite minimization kernel. A finite lexicographic rational objective
can be ranked by Nat; no claim that this is Python's implementation. -/
def argmin {α : Type} (cost : α → Nat) : List α → Option α
  | [] => none
  | a :: rest => match argmin cost rest with
    | none => some a
    | some b => if cost a ≤ cost b then some a else some b

theorem argmin_none {α : Type} (cost : α → Nat) (xs : List α) :
    argmin cost xs = none ↔ xs = [] := by
  cases xs with
  | nil => simp [argmin]
  | cons a rest =>
      simp only [argmin]
      cases argmin cost rest <;> simp
      split <;> simp

theorem argmin_spec {α : Type} (cost : α → Nat) (xs : List α) (best : α)
    (hb : argmin cost xs = some best) :
    best ∈ xs ∧ ∀ x, x ∈ xs → cost best ≤ cost x := by
  induction xs generalizing best with
  | nil => simp [argmin] at hb
  | cons a rest ih =>
      simp only [argmin] at hb
      cases hr : argmin cost rest with
      | none =>
          rw [hr] at hb
          have empty := (argmin_none cost rest).mp hr
          subst rest
          simp at hb
          subst best
          simp
      | some b =>
          rw [hr] at hb
          have spec := ih b hr
          by_cases hab : cost a ≤ cost b
          · simp [hab] at hb
            subst best
            constructor
            · simp
            · intro x hx
              rcases List.mem_cons.mp hx with hx | hx
              · subst x; exact Nat.le_refl _
              · exact Nat.le_trans hab (spec.2 x hx)
          · simp [hab] at hb
            subst best
            constructor
            · exact List.mem_cons_of_mem a spec.1
            · intro x hx
              rcases List.mem_cons.mp hx with hx | hx
              · subst x; omega
              · exact spec.2 x hx

theorem finite_solver_correct {α : Type} (groups : List (List α))
    (feasible : List α → Bool) (cost : List α → Nat) (best : List α)
    (hb : argmin cost ((designs groups).filter feasible) = some best) :
    (Admissible groups best ∧ feasible best = true) ∧
      ∀ x, Admissible groups x → feasible x = true → cost best ≤ cost x := by
  have hs := argmin_spec cost _ best hb
  constructor
  · exact (feasible_filter groups feasible best).mp hs.1
  · intro x ha hf
    exact hs.2 x ((feasible_filter groups feasible x).mpr ⟨ha, hf⟩)

theorem finite_infeasible_correct {α : Type} (groups : List (List α))
    (feasible : List α → Bool) (cost : List α → Nat)
    (hn : argmin cost ((designs groups).filter feasible) = none) :
    ∀ x, Admissible groups x → feasible x ≠ true := by
  have empty := (argmin_none cost _).mp hn
  intro x ha hf
  have hx := (feasible_filter groups feasible x).mpr ⟨ha, hf⟩
  rw [empty] at hx
  exact List.not_mem_nil hx

/- DP resource dominance under a common suffix. Equal response/radius/future
keys must be proved by the caller; these are not inferred from cost alone. -/
structure Resources where
  cost : Rat
  effort : Rat
  count : Nat

def Dominates (a b : Resources) : Prop :=
  a.cost ≤ b.cost ∧ a.effort ≤ b.effort ∧ a.count ≤ b.count

def add (a s : Resources) : Resources :=
  ⟨a.cost+s.cost, a.effort+s.effort, a.count+s.count⟩

theorem dominance_suffix (a b s : Resources) (h : Dominates a b) :
    Dominates (add a s) (add b s) := by
  rcases h with ⟨hc, he, hn⟩
  simp only [Dominates, add]
  constructor
  · grind
  constructor
  · grind
  · omega

theorem dominance_feasible (a b s cap : Resources) (h : Dominates a b)
    (hb : Dominates (add b s) cap) : Dominates (add a s) cap := by
  have hs := dominance_suffix a b s h
  rcases hs with ⟨hc, he, hn⟩
  rcases hb with ⟨bc, be, bn⟩
  simp only [Dominates] at *
  constructor
  · grind
  constructor
  · grind
  · omega

/- Abstract response-DP invariant: every exact prefix has a retained no-worse
representative. These theorems formalize layer composition and final selection;
the acoustic/conflict-key instantiation remains a separate mathematical proof. -/
def Covers {α : Type} (rep : α → α → Prop) (full reduced : List α) : Prop :=
  ∀ x, x ∈ full → ∃ y, y ∈ reduced ∧ rep y x

theorem compression_step {α : Type} (rep : α → α → Prop)
    (full reduced : List α) (expand : α → List α)
    (cover : Covers rep full reduced)
    (simulation : ∀ a b, rep a b → ∀ x, x ∈ expand b →
      ∃ y, y ∈ expand a ∧ rep y x) :
    Covers rep (full.flatMap expand) (reduced.flatMap expand) := by
  intro x hx
  rcases List.mem_flatMap.mp hx with ⟨b, hb, hxb⟩
  rcases cover b hb with ⟨a, ha, hab⟩
  rcases simulation a b hab x hxb with ⟨y, hy, hyx⟩
  exact ⟨y, List.mem_flatMap.mpr ⟨a, ha, hy⟩, hyx⟩

theorem compression_trans {α : Type} (rep : α → α → Prop)
    (full middle reduced : List α)
    (trans : ∀ a b c, rep a b → rep b c → rep a c)
    (first : Covers rep full middle) (second : Covers rep middle reduced) :
    Covers rep full reduced := by
  intro x hx
  rcases first x hx with ⟨y, hy, hyx⟩
  rcases second y hy with ⟨z, hz, hzy⟩
  exact ⟨z, hz, trans z y x hzy hyx⟩

def SafeRep {α : Type} (feasible : α → Bool) (cost : α → Nat) (a b : α) : Prop :=
  (feasible b = true → feasible a = true) ∧ cost a ≤ cost b

theorem compressed_optimal {α : Type} (full reduced : List α)
    (feasible : α → Bool) (cost : α → Nat)
    (cover : Covers (SafeRep feasible cost) full reduced)
    (subset : ∀ y, y ∈ reduced → y ∈ full) (best : α)
    (hb : argmin cost (reduced.filter feasible) = some best) :
    (best ∈ full ∧ feasible best = true) ∧
      ∀ x, x ∈ full → feasible x = true → cost best ≤ cost x := by
  have hs := argmin_spec cost _ best hb
  have hm := List.mem_filter.mp hs.1
  constructor
  · exact ⟨subset best hm.1, hm.2⟩
  · intro x hx hf
    rcases cover x hx with ⟨y, hy, hrep⟩
    have hfy := hrep.1 hf
    have hby := hs.2 y (List.mem_filter.mpr ⟨hy, hfy⟩)
    exact Nat.le_trans hby hrep.2

theorem compressed_infeasible {α : Type} (full reduced : List α)
    (feasible : α → Bool) (cost : α → Nat)
    (cover : Covers (SafeRep feasible cost) full reduced)
    (hn : argmin cost (reduced.filter feasible) = none) :
    ∀ x, x ∈ full → feasible x ≠ true := by
  have empty := (argmin_none cost _).mp hn
  intro x hx hf
  rcases cover x hx with ⟨y, hy, hrep⟩
  have hmember := List.mem_filter.mpr ⟨hy, hrep.1 hf⟩
  rw [empty] at hmember
  exact List.not_mem_nil hmember

/- Single-frequency projection certificate for multi-frequency intersection feasibility.
If every candidate satisfies a property at a single projection frequency at most k times,
the joint conjunction cannot hold more than k times. -/
theorem projection_certificate_upper_bound {α : Type} (candidates : List α)
    (f_pass : α → Nat) (all_pass : α → Nat) (k : Nat)
    (h_bound : ∀ x ∈ candidates, all_pass x ≤ f_pass x)
    (h_proj : ∀ x ∈ candidates, f_pass x ≤ k) :
    ∀ x ∈ candidates, all_pass x ≤ k := by
  intro x hx
  exact Nat.le_trans (h_bound x hx) (h_proj x hx)

theorem projection_infeasible {α : Type} (candidates : List α)
    (f_pass : α → Nat) (all_pass : α → Nat) (target k : Nat)
    (h_bound : ∀ x ∈ candidates, all_pass x ≤ f_pass x)
    (h_proj : ∀ x ∈ candidates, f_pass x ≤ k)
    (h_lt : k < target) :
    ∀ x ∈ candidates, ¬ (target ≤ all_pass x) := by
  intro x hx h_ge
  have h_le := projection_certificate_upper_bound candidates f_pass all_pass k h_bound h_proj x hx
  have h_contr := Nat.lt_of_le_of_lt h_le (Nat.lt_of_lt_of_le h_lt h_ge)
  exact Nat.lt_irrefl (all_pass x) h_contr

#print axioms enumeration_complete
#print axioms finite_solver_correct
#print axioms finite_infeasible_correct
#print axioms pruning_sound
#print axioms lower_envelope_prune
#print axioms upper_envelope_prune
#print axioms quota_prune
#print axioms dominance_suffix
#print axioms dominance_feasible
#print axioms compression_step
#print axioms compression_trans
#print axioms compressed_optimal
#print axioms compressed_infeasible
#print axioms projection_certificate_upper_bound
#print axioms projection_infeasible
end MiniBass
