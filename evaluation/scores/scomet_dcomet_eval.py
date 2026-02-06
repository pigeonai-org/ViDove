"""
COMET evaluation module containing both standard COMET (sCOMET) and 
directional COMET (dCOMET) implementations
"""

from comet import download_model, load_from_checkpoint

class CometEvaluator:
    def __init__(self):
        # Initialize standard COMET model
        self.scomet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        
        # Initialize directional COMET (dCOMET) model
        # The model used by DELTA: https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-mqm.tar.gz
        self.dcomet_model = load_from_checkpoint(download_model("NataliaKhaidanova/wmt21-comet-qe-mqm"))
    
    def evaluate_scomet(self, src, mt, ref, batch_size=1, use_gpu=False):
        """
        Evaluate translation using standard COMET (requires reference)
        
        Args:
            src: Source text
            mt: Machine translation
            ref: Reference translation
            batch_size: Batch size for prediction
            use_gpu: Whether to use GPU for prediction
            
        Returns:
            COMET score
        """
        comet_output = self.scomet_model.predict(
            [{"src": src, "mt": mt, "ref": ref}], 
            batch_size=batch_size, 
            gpus=0 if not use_gpu else 1
        )
        return comet_output.scores[0]
    
    def evaluate_dcomet(self, src, mt, batch_size=1, use_gpu=False):
        """
        Evaluate translation using directional COMET (reference-free)
        
        Args:
            src: Source text
            mt: Machine translation
            batch_size: Batch size for prediction
            use_gpu: Whether to use GPU for prediction
            
        Returns:
            dCOMET score
        """
        dcomet_output = self.dcomet_model.predict(
            [{"src": src, "mt": mt}], 
            batch_size=batch_size, 
            gpus=0 if not use_gpu else 1
        )
        return dcomet_output.scores[0]
    
    def evaluate_batch_scomet(self, srcs, mts, refs, batch_size=1, use_gpu=False):
        """
        Batch evaluate translations using standard COMET
        
        Args:
            srcs: List of source texts
            mts: List of machine translations
            refs: List of reference translations
            batch_size: Batch size for prediction
            use_gpu: Whether to use GPU for prediction
            
        Returns:
            List of COMET scores
        """
        samples = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, mts, refs)]
        comet_output = self.scomet_model.predict(samples, batch_size=batch_size, gpus=0 if not use_gpu else 1)
        return comet_output.scores
    
    def evaluate_batch_dcomet(self, srcs, mts, batch_size=1, use_gpu=False):
        """
        Batch evaluate translations using directional COMET
        
        Args:
            srcs: List of source texts
            mts: List of machine translations
            batch_size: Batch size for prediction
            use_gpu: Whether to use GPU for prediction
            
        Returns:
            List of dCOMET scores
        """
        samples = [{"src": src, "mt": mt} for src, mt in zip(srcs, mts)]
        dcomet_output = self.dcomet_model.predict(samples, batch_size=batch_size, gpus=0 if not use_gpu else 1)
        return dcomet_output.scores 