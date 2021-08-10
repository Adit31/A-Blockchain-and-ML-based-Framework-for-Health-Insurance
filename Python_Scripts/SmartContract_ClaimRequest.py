import smartpy as sp

class Claims(sp.Contract):
    def __init__(self, CoverageAmount):
        self.init(
        TotalClaimAmount = 0,
        PolicyCoverage = CoverageAmount
        )

    @sp.entry_point
    def CheckClaim(self, ClaimAmount):
        sp.verify(self.data.TotalClaimAmount + ClaimAmount <= self.data.PolicyCoverage, message = "Limit Exceeded, Claim Cannot be Processed")

        self.data.TotalClaimAmount += ClaimAmount

@sp.add_test(name = "Claim Verification")
def test():
    scenario = sp.test_scenario()
    CoverageAmount = sp.nat(1000)
    ClaimRequested = sp.nat(100)

    SmartContract = Claims(CoverageAmount = CoverageAmount)
    scenario += SmartContract

    scenario += SmartContract.CheckClaim(ClaimRequested).run()
    scenario += SmartContract.CheckClaim(ClaimRequested).run()

    ClaimRequested = sp.nat(1000)
    scenario += SmartContract.CheckClaim(ClaimRequested).run(valid = False)